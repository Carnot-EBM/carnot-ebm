old-code-trace-repair-hard-0015-1.
old-code-trace-repair-hard-0015-1.
assert binary_select_leftmost([1, 2, 2, 2, 5], 2) == 1
assert binary_search_most_right_of_target(binary_search_leftmost([1,  old_index, old_index, old_index, 5], 2)) == 2
...
(Note: The baseline candidate is actually a O(N) linear search, linear search is not a "binary search" despite the name. However, the requirements state "Return the leftmost index of target in a sorted list, or -1." and "binary_search_leftmost" is the input function. The functional requirement is the "leftmost index". "binary_search_leftmost" is the name, and[] is the target. The functional requirement is the "expected behavior" is "Return the leftmost index of target in a sorted list, or -1." and "expected behavior: Return the leftmost index of target in a sorted list, or -1."

The baseline candidate is O(N) and returns the *rightmost* index because it continues the loop.
The failing assertion is: `assert binary_search_leftmost([1, 2, 2, 2, 5], 2) == 1`.
The baseline candidate returns `3` for `[