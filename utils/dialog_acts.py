from typing import Dict

from core.game import PersuasionGame, NegotiationGame


SYSTEM_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.S_Greeting: "The Persuader greets the Persuadee to open or maintain the conversation politely.",
	PersuasionGame.S_CredibilityAppeal: "The Persuader provides facts, evidence, or reputation signals to establish the charity's credibility.",
	PersuasionGame.S_EmotionAppeal: "The Persuader uses emotional language or stories to inspire empathy and motivate support.",
	PersuasionGame.S_PropositionOfDonation: "The Persuader explicitly asks the Persuadee to make a donation or take the next step toward donating.",
	PersuasionGame.S_LogicalAppeal: "The Persuader uses reasoning, benefits, or cause-and-effect logic to justify donating.",
	PersuasionGame.S_TaskRelatedInquiry: "The Persuader asks questions to understand the Persuadee's knowledge, preferences, or constraints.",
	PersuasionGame.S_Other: "The Persuader responds without using any specific persuasion strategy listed above.",
}


USER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	PersuasionGame.U_NoDonation: "The Persuadee declines or refuses to donate, or states they will not donate.",
	PersuasionGame.U_NegativeReaction: "The Persuadee reacts negatively, expresses doubt, or raises objections without clearly refusing.",
	PersuasionGame.U_Neutral: "The Persuadee remains undecided, neutral, or requests more information without showing clear sentiment.",
	PersuasionGame.U_PositiveReaction: "The Persuadee reacts positively or favorably but does not explicitly commit to donating.",
	PersuasionGame.U_Donate: "The Persuadee explicitly agrees or commits to donating.",
}


SELLER_DIALOG_ACT_DEFINITIONS: Dict[str, str] = {
	NegotiationGame.S_INTRO: "Open the Craigslist conversation with a polite greeting or reference to the listing.",
	NegotiationGame.S_INIT_PRICE: "State the initial asking price or remind the buyer of the listed amount.",
	NegotiationGame.S_INFORM: "Provide factual details about the item’s condition, history, or included accessories.",
	NegotiationGame.S_OFFER: "Propose a new price or bundle that could close the deal quickly.",
	NegotiationGame.S_COUNTER: "Counter the buyer’s previous price suggestion while keeping the discussion friendly.",
	NegotiationGame.S_VAGUE: "Answer price questions indirectly when you want to keep flexibility or gauge interest.",
	NegotiationGame.S_INSIST: "Hold firm on a price point or reiterate that the current offer is already fair.",
	NegotiationGame.S_ACCEPT: "Explicitly accept the buyer’s latest price or condition to close the negotiation.",
	NegotiationGame.S_REJECT: "Decline the buyer’s terms or explain why the offer cannot be accepted.",
	NegotiationGame.S_QUIT: "Politely exit the negotiation when continuing no longer makes sense.",
	NegotiationGame.S_OTHER: "Respond in any other way that keeps the conversation civil but doesn’t fit specific tactics.",
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
