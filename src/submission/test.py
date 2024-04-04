import random

block_size = 100
x = "This is a test I created to check if the rearrange looks correct. Adding more words to see what happens with a longer input"

trunc_len = random.randint(
    4, min(int(block_size*3/4), len(x)))
start_idx = random.randint(0, len(x) - trunc_len)
print(start_idx)
truncated_document = x[start_idx:start_idx+trunc_len]

print(f'{truncated_document} with length {len(truncated_document)}')

# 2.
avg_mask_len = round(len(truncated_document) * 0.25)
variance = round(avg_mask_len * 0.25)
print(f'avg masked len: {avg_mask_len} and variance: {variance}')
masked_len = random.randint(
    avg_mask_len - variance, avg_mask_len + variance)
print(f'masked len {masked_len}')
start_idx_masked = random.randint(
    1, len(truncated_document) - masked_len - 1)
print(f'start index masked: {start_idx_masked}')

prefix = truncated_document[:start_idx_masked]
masked_content = truncated_document[start_idx_masked:start_idx_masked+masked_len]
suffix = truncated_document[start_idx_masked+masked_len:]

print(prefix)
print(masked_content)
print(suffix)


sMASK_CHAR = u"\u2047"  # the doublequestionmark character, for mask
sPAD_CHAR = u"\u25A1"  # the empty square character, for pad
# 3.
masked_string = prefix + sMASK_CHAR + suffix + \
    sMASK_CHAR + masked_content + sMASK_CHAR

print(masked_string)
masked_string = masked_string + sPAD_CHAR * \
    (block_size - (len(masked_string) - 1))
print(masked_string)

# 4.
x, y = masked_string[:-1], masked_string[1:]
print(x)
print(y)


inp = " Mouradian. Khatchig Mouradian⁇t, writer and tr⁇ is a journalis⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□"
print(f'lenght of input: {len(inp)}')

a = "Cristina"
print(a[:4])
print(a[4:7])
print(a[7:])
