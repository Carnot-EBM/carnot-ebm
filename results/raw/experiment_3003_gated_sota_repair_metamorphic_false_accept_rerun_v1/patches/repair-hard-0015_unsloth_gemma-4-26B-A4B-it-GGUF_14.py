def binary_search_leftmost(nums, target):
    low = 0
    high = len(nums) - 1
    leftmost_index = -1
    while low <= high:
        mid = (low + high) // 2
        if nums[mid] == target:
            leftmost_index = mid
            high = mid - 1
        elif nums[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return leftmost_index
