def is_palindrome_text(text):
    filtered_chars = [char.lower() for char in text if char.isalnum()]
    return filtered_chars == filtered_chars[::-1]
