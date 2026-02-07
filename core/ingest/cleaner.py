def clean_text(text: str) -> str:
    return (
        text.replace("\n\n", "\n")
            .replace("Advertisement", "")
            .strip()
    )
