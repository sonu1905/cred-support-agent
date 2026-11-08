"""Task 14 demonstration: approve unchanged + revise bad claim."""

import asyncio

from autogen_review import (
    FinalVerdictMessage,
    ReviewNoteMessage,
    review_answer,
)


GROUNDED = (
    "Some loan agreements permit early repayment without a charge while others "
    "include a prepayment fee. The applicable rule depends on the product and "
    "signed agreement."
)

EVIDENCE = (
    "Prepayment rules vary by product. A fee may apply when specified in the "
    "signed loan agreement."
)


async def run_demo() -> None:
    print("CASE 1 - supported answer should be approved unchanged")
    verdict, messages = await review_answer(
        draft_answer=GROUNDED,
        grounded_answer=GROUNDED,
        evidence=EVIDENCE,
    )
    print("RoundRobin agent messages:")
    for message in messages:
        if isinstance(message, (ReviewNoteMessage, FinalVerdictMessage)):
            print(f"  {message.source}: {message.content}")
    print("Final verdict:", verdict.model_dump())

    print("\nCASE 2 - deliberately injected bad claim should be revised")
    bad_draft = GROUNDED + " Approval is guaranteed for every customer."

    verdict, messages = await review_answer(
        draft_answer=bad_draft,
        grounded_answer=GROUNDED,
        evidence=EVIDENCE,
    )
    print("RoundRobin agent messages:")
    for message in messages:
        if isinstance(message, (ReviewNoteMessage, FinalVerdictMessage)):
            print(f"  {message.source}: {message.content}")
    print("Final verdict:", verdict.model_dump())


if __name__ == "__main__":
    asyncio.run(run_demo())
