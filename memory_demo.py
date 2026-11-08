"""Task 8 transcript: carried state and fresh state."""

from memory_layer import chat


if __name__ == "__main__":
    print("SESSION A")
    first = "Please remember application CRD-0001."
    print("User:", first)
    print("Agent:", chat("session-a", first))

    second = "What is its status?"
    print("User:", second)
    print("Agent:", chat("session-a", second))

    print("\nSESSION B - fresh conversation")
    print("User:", second)
    print("Agent:", chat("session-b", second))
