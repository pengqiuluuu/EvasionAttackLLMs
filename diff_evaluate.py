import difflib


def count_difference(para1, para2):
    # Split the text into lines for comparison
    old_lines = para1.split()
    new_lines = para2.split()

    # Create a Differ object
    d = difflib.Differ()

    # Generate a diff
    diff = list(d.compare(old_lines, new_lines))

    # Count additions and deletions
    additions = sum(1 for line in diff if line.startswith('+ '))
    deletions = sum(1 for line in diff if line.startswith('- '))

    diff_count = additions + deletions
    print("Number of differences: ", diff_count)
    return diff_count


# Evaluate the difference between the original and modified content
avg_count = 0

file_1 = open("data/attack_files/imdb-attacked-100/content.csv", "r")
file_2 = open("data/attack_files/imdb-attacked-100/modified_content.csv", "r")

for i in range(200):
    para1 = file_1.readline()
    if '", 0' in para1:
        para1, _ = para1.strip().split('", 0')
    elif '", 1' in para1:
        para1, _ = para1.strip().split('", 1')
    else:
        pass

    para2 = file_2.readline()
    if '", 0' in para2:
        para2, _ = para2.strip().split('", 0')
    elif '", 1' in para2:
        para2, _ = para2.strip().split('", 1')
    else:
        pass

    avg_count += count_difference(para1.lstrip('"'), para2.lstrip('"'))

print("Average difference count: ", avg_count/200)