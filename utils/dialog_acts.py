from typing import Dict

from core.game import PersuasionGame


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


__all__ = [
	"SYSTEM_DIALOG_ACT_DEFINITIONS",
	"USER_DIALOG_ACT_DEFINITIONS",
]
