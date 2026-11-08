"""Task 14: offline two-agent AutoGen review workflow."""

import json
from typing import Literal, Sequence

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    BaseChatMessage,
    BaseTextChatMessage,
    TextMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core import CancellationToken
from pydantic import BaseModel


class ReviewNote(BaseModel):
    supported: bool
    issue: str


class FinalVerdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str


class ReviewNoteMessage(BaseTextChatMessage):
    """Custom AutoGen message used by the first reviewer."""

    type: Literal["ReviewNoteMessage"] = "ReviewNoteMessage"


class FinalVerdictMessage(BaseTextChatMessage):
    """Custom AutoGen message carrying the final Pydantic verdict as JSON."""

    type: Literal["FinalVerdictMessage"] = "FinalVerdictMessage"


def _task_payload(messages: Sequence[BaseChatMessage]) -> dict:
    for message in messages:
        if isinstance(message, TextMessage) and message.source == "user":
            return json.loads(message.content)

    raise ValueError("Review task payload was not found")


class GroundingReviewerAgent(BaseChatAgent):
    def __init__(self) -> None:
        super().__init__(
            name="grounding_reviewer",
            description="Checks a draft answer against the grounded answer and evidence.",
        )

    @property
    def produced_message_types(self):
        return (ReviewNoteMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        payload = _task_payload(messages)
        draft = payload["draft_answer"].strip()
        grounded = payload["grounded_answer"].strip()

        supported = draft == grounded

        if supported:
            note = ReviewNote(
                supported=True,
                issue="No unsupported claim found.",
            )
        else:
            note = ReviewNote(
                supported=False,
                issue=(
                    "The draft differs from the grounded answer. "
                    "Remove the unsupported addition and restore the grounded answer."
                ),
            )

        message = ReviewNoteMessage(
            content=note.model_dump_json(),
            source=self.name,
        )
        return Response(chat_message=message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return None


class FinalEditorAgent(BaseChatAgent):
    def __init__(self) -> None:
        super().__init__(
            name="final_editor",
            description="Produces the final structured approval or revision verdict.",
        )

    @property
    def produced_message_types(self):
        return (FinalVerdictMessage,)

    async def on_messages(
        self,
        messages: Sequence[BaseChatMessage],
        cancellation_token: CancellationToken,
    ) -> Response:
        payload = _task_payload(messages)

        review_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, ReviewNoteMessage)
            ),
            None,
        )

        if review_message is None:
            raise ValueError("Reviewer message was not received")

        review = ReviewNote.model_validate_json(review_message.content)

        if review.supported:
            verdict = FinalVerdict(
                approved=True,
                final_answer=payload["draft_answer"],
                reason="Reviewer found no unsupported claim; answer kept unchanged.",
            )
        else:
            verdict = FinalVerdict(
                approved=False,
                final_answer=payload["grounded_answer"],
                reason=(
                    "Reviewer detected an unsupported claim; final editor "
                    "replaced the draft with the grounded answer."
                ),
            )

        message = FinalVerdictMessage(
            content=verdict.model_dump_json(),
            source=self.name,
        )
        return Response(chat_message=message)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        return None


def build_review_team() -> RoundRobinGroupChat:
    reviewer = GroundingReviewerAgent()
    editor = FinalEditorAgent()

    return RoundRobinGroupChat(
        participants=[reviewer, editor],
        max_turns=2,
        custom_message_types=[ReviewNoteMessage, FinalVerdictMessage],
    )


async def review_answer(
    *,
    draft_answer: str,
    grounded_answer: str,
    evidence: str,
) -> tuple[FinalVerdict, list[BaseChatMessage]]:
    payload = {
        "draft_answer": draft_answer,
        "grounded_answer": grounded_answer,
        "evidence": evidence,
    }

    team = build_review_team()

    result = await team.run(
        task=TextMessage(
            content=json.dumps(payload),
            source="user",
        )
    )

    verdict_message = next(
        (
            message
            for message in reversed(result.messages)
            if isinstance(message, FinalVerdictMessage)
        ),
        None,
    )

    if verdict_message is None:
        raise RuntimeError("Final editor did not produce a verdict")

    verdict = FinalVerdict.model_validate_json(verdict_message.content)
    return verdict, list(result.messages)
