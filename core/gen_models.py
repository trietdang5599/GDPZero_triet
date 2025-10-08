import requests
import logging
import torch
import os
import multiprocessing as mp
import nltk
import re

from pathlib import Path
from openai import OpenAI, AzureOpenAI

from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, set_seed
from typing import List, Tuple, Dict, Optional
from core.helpers import DialogSession
from functools import lru_cache
from tenacity import retry, stop_after_attempt,	wait_exponential, wait_fixed  # for exponential backoff
from utils.utils import hashabledict


logger = logging.getLogger(__name__)


_ENV_VARS_LOADED = False


def _load_env_vars() -> None:
	"""Populate `os.environ` with keys from the project .env file if not already set."""
	global _ENV_VARS_LOADED
	if _ENV_VARS_LOADED:
		return
	env_path = Path(__file__).resolve().parents[1] / ".env"
	if not env_path.exists():
		logger.debug(".env file not found at %s", env_path)
		_ENV_VARS_LOADED = True
		return
	try:
		with env_path.open("r", encoding="utf-8") as env_file:
			for raw_line in env_file:
				line = raw_line.strip()
				if not line or line.startswith("#") or "=" not in line:
					continue
				key, value = line.split("=", 1)
				key = key.strip()
				value = value.split("#", 1)[0].strip().strip("\"'")
				if key and key not in os.environ:
					os.environ[key] = value
	except Exception as exc:
		logger.warning("Failed to load environment variables from %s: %s", env_path, exc)
	finally:
		_ENV_VARS_LOADED = True


_load_env_vars()


@lru_cache(maxsize=None)
def _get_openai_client(api_key: str) -> OpenAI:
	if not api_key:
		raise ValueError("OPENAI_API_KEY is not set; please configure it in the environment or .env file.")
	return OpenAI(api_key=api_key)


@lru_cache(maxsize=None)
def _get_azure_openai_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAI:
	if not api_key:
		raise ValueError("MS_OPENAI_API_KEY is not set; please configure it in the environment or .env file.")
	if not endpoint:
		raise ValueError("MS_OPENAI_API_BASE is not set; please configure it in the environment or .env file.")
	return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)


class GenerationModel(ABC):
	# used to generate text in general. e.g. could be using API, or local model
	@abstractmethod
	def generate(self, input_text, **gen_args):
		"""
		Generate text from the model.
		"""
		raise NotImplementedError

	def chat_generate(self, messages, **gen_args):
		"""
		Generate text from the model. Used for chatbot.
		"""
		raise NotImplementedError
	
	def chat_generate_batched(self, messages_list, **gen_args):
		"""
		Generate text from the model when you have multiple message histories
		"""
		raise NotImplementedError

	def _cleaned_resp(self, data, prompt) -> "List[str]":
		# default helper function to clean extract the generated text from the returned json
		logger.debug("promopt:")
		logger.debug(prompt)
		cleaned_resps = []
		for gen_resp in data:
			logger.debug("raw response:")
			logger.debug(gen_resp['generated_text'])
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			logger.debug(f"cleaned response: {cleaned_resp}")
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps
	
	def _cleaned_chat_resp(self, data, assistant_role="Persuader:", user_role="Persuadee:") -> "List[str]":
		# remove the user_role and keep the assistant_role
		# default helper function to clean extract the generated text from the returned json
		cleaned_resps = []
		for gen_resp in data:
			logger.debug("raw response:")
			logger.debug(gen_resp['generated_text'])
			cleaned_resp = gen_resp['generated_text'].strip()
			if "\n" in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index("\n")]
			if assistant_role in cleaned_resp:
				cleaned_resp = cleaned_resp[cleaned_resp.index(assistant_role) + len(assistant_role):].strip()
			if user_role in cleaned_resp:
				cleaned_resp = cleaned_resp[:cleaned_resp.index(user_role)].strip()
			logger.debug(f"cleaned response: {cleaned_resp}")
			cleaned_resps.append(cleaned_resp)
		return cleaned_resps


class DialogModel(ABC):
	# used to play DialogGame
	def __init__(self):
		self.dialog_acts = []
		return
	
	@abstractmethod
	def get_utterance(self, state:DialogSession, action) -> str:
		raise NotImplementedError
	
	def get_utterance_batched(self, state:DialogSession, action:int, batch:int) -> List[str]:
		raise NotImplementedError

	@abstractmethod
	def get_utterance_w_da(self, state:DialogSession, action) -> Tuple[str, str]:
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
	
	def get_utterance_w_da_from_batched_states(self, states:List[DialogSession], action=None):
		# this is used for user agent. should not be used for system agent
		raise NotImplementedError
		


class APIModel(GenerationModel):
	API_TOKEN = os.environ.get("HF_API_KEY")

	def __init__(self):
		# self.API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
		self.API_URL = "https://api-inference.huggingface.co/models/gpt2-large"
		self.headers: dict[str, str] = {"Authorization": f"Bearer {APIModel.API_TOKEN}"}
		self.inference_args = {
			"max_new_tokens": 100,
			"temperature": 0.7,
			"repetition_penalty": 1.2,
			"return_full_text": False
		}
		return

	def generate(self, input_text, **_args):
		data = {
			"inputs": input_text,
			"parameters": _args or self.inference_args
		}
		response = requests.post(self.API_URL, headers=self.headers, json=data)
		return response.json()


class OpenAIModel(GenerationModel):
	API_TOKEN = os.environ.get("OPENAI_API_KEY")

	def __init__(self, model_name="text-curie-001"):
		self.model_name = model_name
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"echo": False,
			"n": 1,
			"stop": "\n",
		}
		try:
			available = {model.id for model in _get_openai_client(self.API_TOKEN).models.list().data}
			if model_name not in available:
				logger.warning("OpenAI model %s not available for provided API key", model_name)
		except Exception as exc:
			logger.debug("Skipping OpenAI model availability check due to: %s", exc)
		return

	def _update_args(self, new_args):
		args = {**self.inference_args}
		from_cache = False
		if "max_new_tokens" in new_args:
			new_args["max_tokens"] = new_args.pop("max_new_tokens")
		if "return_full_text" in new_args:
			new_args["echo"] = new_args.pop("return_full_text")
		if "do_sample" in new_args:
			from_cache = not new_args.pop("do_sample")  # rely on caching
		if "num_return_sequences" in new_args:
			new_args["n"] = new_args.pop("num_return_sequences")
		if "repetition_penalty" in new_args:
			new_args["frequency_penalty"] = new_args.pop("repetition_penalty")
		return from_cache, {**args, **new_args}

	@staticmethod
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		client = _get_openai_client(OpenAIModel.API_TOKEN)
		return client.completions.create(**parameters)

	# tried custom implementation of waiting before request, but I think openai is lying about how it calculates the rate limit
	# takes 3 trials to reach 2^3=8. Then 7 * 8 = 56 sec max. Just to safe we wait a bit more than 10 times
	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def generate(self, input_text, **_args):
		from_cache, parameters = self._update_args(_args)
		parameters["prompt"] = input_text
		if from_cache:
			response = OpenAIModel._cached_generate(**parameters)
		else:
			response = _get_openai_client(self.API_TOKEN).completions.create(**parameters)
		gen_output = []
		for resp in response.choices:
			text = getattr(resp, "text", None)
			if text is None and hasattr(resp, "message"):
				text = getattr(resp.message, "content", "")
			gen_output.append({"generated_text": text or ""})
		return gen_output


class OpenAIChatModel(OpenAIModel):
	def __init__(self, model_name="gpt-3.5-turbo", gen_sentences=-1):
		self.model_name = model_name
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
		}
		try:
			available = {model.id for model in _get_openai_client(self.API_TOKEN).models.list().data}
			if model_name not in available:
				logger.warning("OpenAI chat model %s not available for provided API key", model_name)
		except Exception as exc:
			logger.debug("Skipping OpenAI chat model availability check due to: %s", exc)
		return

	def _update_args(self, new_args):
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		return super()._update_args(new_args)

	def generate(self, input_text, **_args):
		logging.info("It is recommended to use chat_generate instead of generate for OpenAIChatModel")
		messages = [{
			"role": "user",
			"content": input_text
		}]
		return self.chat_generate(messages, **_args)

	@staticmethod
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		client = _get_openai_client(OpenAIChatModel.API_TOKEN)
		parameters = dict(parameters)
		parameters["messages"] = list(parameters["messages"])
		return client.chat.completions.create(**parameters)

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		# generate in a chat format
		from_cache, parameters = self._update_args(gen_args)
		hashable_messages = [hashabledict(m) for m in messages]
		parameters["messages"] = hashable_messages
		if from_cache:
			parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
			response = OpenAIChatModel._cached_generate(**parameters)
		else:
			parameters_for_call = dict(parameters)
			parameters_for_call["messages"] = messages
			response = _get_openai_client(self.API_TOKEN).chat.completions.create(**parameters_for_call)
		gen_output = []
		for resp in response.choices:
			text = getattr(resp.message, "content", "")
			if self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})
		return gen_output
	
	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]


class AzureOpenAIModel(OpenAIModel):
	API_TOKEN = os.environ.get("MS_OPENAI_API_KEY")
	API_BASE = os.environ.get("MS_OPENAI_API_BASE")
	API_VERSION = "2022-12-01"

	def __init__(self, model_name="chatgpt-turbo"):
		self.model_name = model_name
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"echo": False,
			"n": 1,
			"stop": "\n",
		}
		return

	@staticmethod
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		client = _get_azure_openai_client(
			AzureOpenAIModel.API_TOKEN,
			AzureOpenAIModel.API_BASE,
			AzureOpenAIModel.API_VERSION,
		)
		return client.completions.create(**parameters)

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def generate(self, input_text, **_args):
		from_cache, parameters = self._update_args(_args)
		parameters["prompt"] = input_text
		if from_cache:
			response = AzureOpenAIModel._cached_generate(**parameters)
		else:
			client = _get_azure_openai_client(self.API_TOKEN, self.API_BASE, self.API_VERSION)
			response = client.completions.create(**parameters)
		gen_output = []
		for resp in response.choices:
			text = getattr(resp, "text", None)
			if text is None and hasattr(resp, "message"):
				text = getattr(resp.message, "content", "")
			gen_output.append({"generated_text": text or ""})
		return gen_output


class AzureOpenAIChatModel(AzureOpenAIModel):
	API_VERSION = "2023-03-15-preview"

	def __init__(self, model_name="chatgpt", gen_sentences=-1):
		self.model_name = model_name
		self.gen_sentences = None if gen_sentences < 0 else gen_sentences
		self.inference_args = {
			"model": model_name,
			"max_tokens": 64,
			"temperature": 0.7,
			"n": 1,
		}
		return
	
	def _update_args(self, new_args):
		if "stop" in new_args:
			new_args.pop("stop")
		if "echo" in new_args:
			new_args.pop("echo")
		if "return_full_text" in new_args:
			new_args.pop("return_full_text")
		return super()._update_args(new_args)
	
	@staticmethod
	@lru_cache(maxsize=None)
	def _cached_generate(**parameters):
		client = _get_azure_openai_client(
			AzureOpenAIChatModel.API_TOKEN,
			AzureOpenAIChatModel.API_BASE,
			AzureOpenAIChatModel.API_VERSION,
		)
		parameters = dict(parameters)
		parameters["messages"] = list(parameters["messages"])
		return client.chat.completions.create(**parameters)

	@retry(wait=wait_exponential(multiplier=2, min=2, max=8), stop=stop_after_attempt(15))
	def chat_generate(self, messages: List[Dict], **gen_args):
		# generate in a chat format
		from_cache, parameters = self._update_args(gen_args)
		hashable_messages = [hashabledict(m) for m in messages]
		parameters["messages"] = hashable_messages
		if from_cache:
			parameters["messages"] = tuple(hashable_messages)  # list cannot be hashed, so cannot do **parameters
			response = AzureOpenAIChatModel._cached_generate(**parameters)
		else:
			parameters_for_call = dict(parameters)
			parameters_for_call["messages"] = messages
			client = _get_azure_openai_client(self.API_TOKEN, self.API_BASE, self.API_VERSION)
			response = client.chat.completions.create(**parameters_for_call)
		
		# format to a common format
		gen_output = []
		for resp in response.choices:
			text = getattr(resp.message, "content", "")
			if self.gen_sentences is not None:
				sentences = nltk.sent_tokenize(text)
				if len(sentences) > self.gen_sentences:
					text = " ".join(sentences[:self.gen_sentences])
			gen_output.append({"generated_text": text})
		return gen_output
	
	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		pool = mp.Pool(processes=len(messages_list))
		results = []
		for messages in messages_list:
			results.append(pool.apply_async(self.chat_generate, args=(messages,), kwds=gen_args))
		pool.close()
		pool.join()
		return [r.get() for r in results]
	
	def generate(self, input_text, **_args):
		messages = [{
			"role": "user",
			"content": input_text
		}]
		return self.chat_generate(messages, **_args)


class LocalModel(GenerationModel):
	def __init__(
		self,
		model_name: str = "gpt2",
		input_max_len: int = 512,
		stop_symbol: str = "\n",
		cuda: bool = True,
		trust_remote_code: bool = False,
		model_kwargs: Optional[Dict] = None,
	):
		self.device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
		self.cuda = self.device.type == "cuda"
		load_kwargs = model_kwargs.copy() if model_kwargs else {}
		self.tokenizer = AutoTokenizer.from_pretrained(
			model_name,
			truncation_side="left",
			trust_remote_code=trust_remote_code,
		)
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
		if model_name.startswith("meta-llama") or model_name.startswith("Meta-Llama"):
			load_kwargs["load_in_8bit"] = self.cuda
			self.model = LlamaForCausalLM.from_pretrained(
				model_name,
				trust_remote_code=trust_remote_code,
				local_files_only=True,
				**load_kwargs,	
			)
		else:
			self.model = AutoModelForCausalLM.from_pretrained(
				model_name,
				trust_remote_code=trust_remote_code,
				**load_kwargs,
			)
		self.model.to(self.device)
		self.model.eval()
		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		stop_token_ids = self.tokenizer.encode(stop_symbol, add_special_tokens=False)
		self.stop_token_id = stop_token_ids[-1] if len(stop_token_ids) > 0 else self.tokenizer.eos_token_id
		set_seed(42)
		self.input_max_len = input_max_len
		self.default_chat_prefixes = {"assistant": "Assistant:", "user": "User:"}
		self.inference_args = {
			"max_new_tokens": 128,
			"temperature": 0.7,
			"repetition_penalty": 1.0,
			"do_sample": True,
			"num_return_sequences": 1,
			"pad_token_id": self.tokenizer.pad_token_id,
			"eos_token_id": self.stop_token_id,
			"no_repeat_ngram_size": 3,
		}

	@staticmethod
	def _deduplicate_text(text: str, max_sentence_repeats: int = 2) -> str:
		sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
		deduped: List[str] = []
		repeats: Dict[str, int] = {}
		for sent in sentences:
			normalized = sent.lower()
			repeats.setdefault(normalized, 0)
			if repeats[normalized] >= max_sentence_repeats:
				continue
			repeats[normalized] += 1
			if deduped and normalized == deduped[-1].lower():
				continue
			deduped.append(sent)
		if not deduped:
			candidate = text
		else:
			candidate = " ".join(deduped)
		return LocalModel._trim_repeated_phrases(candidate)

	@staticmethod
	def _trim_repeated_phrases(text: str, max_phrase: int = 8) -> str:
		tokens = text.split()
		limit = len(tokens)
		for window in range(1, min(max_phrase, limit // 2) + 1):
			i = 0
			while i + 2 * window <= limit:
				segment = tokens[i:i + window]
				next_segment = tokens[i + window:i + 2 * window]
				if segment == next_segment:
					return " ".join(tokens[:i + window])
				i += 1
		return " ".join(tokens)

	def _prepare_generation_args(self, gen_args: Dict) -> Dict:
		gen_params = {**self.inference_args}
		gen_params.update(gen_args)
		for legacy_key in ("return_full_text", "stop", "echo", "n", "max_tokens", "logprobs"):
			gen_params.pop(legacy_key, None)
		if gen_params.get("num_return_sequences", 1) < 1:
			gen_params["num_return_sequences"] = 1
		if gen_params.get("num_return_sequences", 1) > 1 and not gen_params.get("do_sample", False):
			gen_params["do_sample"] = True
		if gen_params.get("pad_token_id") is None:
			gen_params["pad_token_id"] = self.tokenizer.pad_token_id
		if gen_params.get("eos_token_id") is None:
			gen_params["eos_token_id"] = self.stop_token_id
		return gen_params

	def _messages_to_prompt(self, messages: List[Dict]) -> str:
		if not messages:
			return ""
		chat_template = getattr(self.tokenizer, "chat_template", None)
		if hasattr(self.tokenizer, "apply_chat_template") and chat_template:
			try:
				return self.tokenizer.apply_chat_template(
					messages,
					tokenize=False,
					add_generation_prompt=True,
				)
			except Exception as exc:
				logger.debug(f"chat template fallback due to: {exc}")
		prompt_lines: List[str] = []
		assistant_prefix = None
		user_prefix = None
		for message in messages:
			content = message.get("content", "").strip()
			if not content:
				continue
			prompt_lines.append(content)
			colon_idx = content.find(":")
			if colon_idx != -1:
				prefix = content[:colon_idx + 1]
				if message.get("role") == "assistant":
					assistant_prefix = prefix
				elif message.get("role") == "user":
					user_prefix = prefix
		last_role = messages[-1].get("role")
		if last_role == "user":
			next_prefix = assistant_prefix or self.default_chat_prefixes["assistant"]
		elif last_role == "assistant":
			next_prefix = user_prefix or self.default_chat_prefixes["user"]
		else:
			next_prefix = assistant_prefix or self.default_chat_prefixes["assistant"]
		prompt_lines.append(f"{next_prefix} ")
		return "\n".join(prompt_lines)

	def generate(self, input_text: str, **gen_args):
		gen_params = self._prepare_generation_args(gen_args)
		inputs = self.tokenizer([input_text], return_tensors='pt', truncation=True, max_length=self.input_max_len)
		inputs = {k: v.to(self.device) for k, v in inputs.items()}
		prompt_len = inputs['input_ids'].shape[-1]
		with torch.no_grad():
			outputs = self.model.generate(**inputs, **gen_params)
		gen_only_outputs = outputs[:, prompt_len:].detach().cpu()
		gen_resps = self.tokenizer.batch_decode(gen_only_outputs, skip_special_tokens=True)
		gen_output = []
		for resp in gen_resps:
			cleaned = self._deduplicate_text(resp)
			gen_output.append({"generated_text": cleaned})
		return gen_output

	def chat_generate(self, messages: List[Dict], **gen_args):
		prompt = self._messages_to_prompt(messages)
		return self.generate(prompt, **gen_args)

	def chat_generate_batched(self, messages_list: List[List[Dict]], **gen_args):
		return [self.chat_generate(messages, **gen_args) for messages in messages_list]
