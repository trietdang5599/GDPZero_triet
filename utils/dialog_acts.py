from typing import Dict

from core.game import PersuasionGame, NegotiationGame


SYSTEM_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.S_Greeting: "Please start or maintain the conversation politely to keep the dialogue friendly.",
	PersuasionGame.S_CredibilityAppeal: "Please use credentials and cite organizational impacts to establish credibility and earn the user's trust. The information usually comes from an objective source such as the organization's website or other well-established websites.",
	PersuasionGame.S_EmotionAppeal: "Please elicit specific emotions to influence the persuadee.",
	PersuasionGame.S_PropositionOfDonation: "Please explicitly invite the persuadee to donate or take the next concrete step toward donating.",
	PersuasionGame.S_LogicalAppeal: "Please use reasoning and evidence to convince the persuadee.",
	PersuasionGame.S_FootInDoor: "Please use the strategy of starting with small donation requests to facilitate compliance followed by larger requests.",
	PersuasionGame.S_SelfModeling: "Please use the self-modeling strategy where you first indicate the persuadee's own intention to donate and choose to act as a role model for the persuadee to follow.",
	PersuasionGame.S_PersonalStory: "Please use narrative exemplars to illustrate someone's donation experiences or the beneficiaries' positive outcomes, which can motivate others to follow the actions.",
	PersuasionGame.S_DonationInformation: "Please provide specific information about the donation task, such as the donation procedure, donation range, and logistics. By providing detailed action guidance, this strategy can enhance the persuadee's self-efficacy and facilitate behavior compliance.",
	PersuasionGame.S_SourceRelatedInquiry: "Please ask if the persuadee is aware of the organization (i.e., the source in our specific donation task).",
	PersuasionGame.S_TaskRelatedInquiry: "Please ask about the persuadee's opinion and expectation related to the task, such as their interests in knowing more about the organization.",
	PersuasionGame.S_PersonalRelatedInquiry: "Please ask about the persuadee's previous personal experiences relevant to charity donation.",
	PersuasionGame.S_Other: "Please respond naturally when no specific persuasion strategy applies.",
}


USER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.U_NoDonation: "The Persuadee declines or refuses to donate, or states they will not donate.",
	PersuasionGame.U_NegativeReaction: "The Persuadee reacts negatively, expresses doubt, or raises objections without clearly refusing.",
	PersuasionGame.U_Neutral: "The Persuadee remains undecided, neutral, or requests more information without showing clear sentiment.",
	PersuasionGame.U_PositiveReaction: "The Persuadee reacts positively or favorably but does not explicitly commit to donating.",
	PersuasionGame.U_Donate: "The Persuadee explicitly agrees or commits to donating.",
}


SELLER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	NegotiationGame.S_GREET: "Please say hello or make small talk to keep the conversation friendly.",
	NegotiationGame.S_ASK: "Please ask any question about the product, year, price, usage, etc.",
	NegotiationGame.S_ANSWER: "Please provide information about the product, year, usage, etc.",
	NegotiationGame.S_FIRST_PRICE: "Please initiate a price or a price range for the product.",
	NegotiationGame.S_COUNTER_PRICE: "Please propose a new price or a new price range.",
	NegotiationGame.S_COMPARATIVE: "Please propose a vague price by using comparatives with the existing price.",
	NegotiationGame.S_CONFIRM_QUESTION: "Please ask a question about the information to be confirmed.",
	NegotiationGame.S_CONFIRM_YES: "Please give an affirmative response to a confirm.",
	NegotiationGame.S_CONFIRM_NO: "Please give a negative response to a confirm.",
	NegotiationGame.S_ACCEPT: "Please agree with the proposed price.",
	NegotiationGame.S_REJECT: "Please disagree with the proposed price.",
	NegotiationGame.S_OTHER: "Please respond naturally when no specific strategy fits.",
}


BUYER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	NegotiationGame.B_GREETING: "Greet the seller or acknowledge their opening message.",
	NegotiationGame.B_INQUIRY: "Ask for clarification about the item’s condition, availability, or logistics before committing.",
	NegotiationGame.B_COUNTER: "Counter the current price with a lower number or new condition.",
	NegotiationGame.B_OFFER: "Propose a concrete offer that differs from the seller’s latest price.",
	NegotiationGame.B_ACCEPT: "Accept the seller’s price or terms and signal readiness to close the deal.",
	NegotiationGame.B_REJECT: "Decline the seller’s terms outright without suggesting alternatives.",
	NegotiationGame.B_QUIT: "End the conversation when you are no longer interested in purchasing.",
	NegotiationGame.B_DISAGREE: "Push back on the seller’s claims or challenge the fairness of the offer.",
	NegotiationGame.B_AGREE: "Verbally agree with the seller’s reasoning while staying open to further discussion.",
	NegotiationGame.B_OTHER: "Respond in any other way that keeps the conversation moving without fitting the above acts.",
}


__all__ = [
	"SYSTEM_DIALOG_ACT_DEFINITIONS",
	"USER_DIALOG_ACT_DEFINITIONS",
	"SELLER_DIALOG_ACT_DEFINITIONS",
	"BUYER_DIALOG_ACT_DEFINITIONS",
]
