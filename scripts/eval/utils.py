# %%
def longest_common_subsequence_length(X, Y):
    m = len(X)
    n = len(Y)
    L = [[0] * (n + 1) for i in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    return L[m][n]


def get_character_level_lcs(generated_text, reference_text, truncate='reference'):

    def normalize_text(text):
        import string
        # Convert to lower case
        lower_case_sentence = text.lower()
        # Split the sentence into words and concatenate them
        concatenated_sentence = ''.join(lower_case_sentence.split())
        # Remove all punctuations
        final_sentence = ''.join(char for char in concatenated_sentence if char not in string.punctuation)
        return final_sentence

    # Tokenize the texts
    generated_tokens = list(normalize_text(generated_text))
    reference_tokens = list(normalize_text(reference_text))

    generated_tokens = generated_tokens[:len(reference_tokens)] if truncate == 'reference' else generated_tokens

    # Compute the longest common subsequence
    lcs_length = longest_common_subsequence_length(generated_tokens, reference_tokens)

    return lcs_length

pythia_tokenizer = None
def get_token_level_lcs(generated_text, reference_text, truncate='reference'):
    global pythia_tokenizer
    if pythia_tokenizer is None:
        from transformers import AutoTokenizer
        pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    
    # Tokenize the texts
    generated_tokens = pythia_tokenizer.encode(generated_text)
    reference_tokens = pythia_tokenizer.encode(reference_text)

    generated_tokens = generated_tokens[:len(reference_tokens)] if truncate == 'reference' else generated_tokens

    # Compute the longest common subsequence
    lcs_length = longest_common_subsequence_length(generated_tokens, reference_tokens)

    return lcs_length


def get_word_level_lcs(generated_text, reference_text, truncate='reference', return_rouge_l=False, return_text=False):
    from nltk import word_tokenize
    
    # Tokenize the texts
    generated_tokens = word_tokenize(generated_text)
    reference_tokens = word_tokenize(reference_text)

    if truncate == 'reference':
        generated_text = generated_tokens[:len(reference_tokens)]
    elif truncate == 'words-50':
        generated_tokens = generated_tokens[:50]
        reference_tokens = reference_tokens[:50]
    else:
        raise ValueError(f"Invalid value for truncate: {truncate}")

    # generated_tokens = generated_tokens[:len(reference_tokens)] if truncate == 'reference' else generated_tokens

    # Compute the longest common subsequence
    lcs_length = longest_common_subsequence_length(generated_tokens, reference_tokens)

    return_tuple = (lcs_length,)
    if return_rouge_l:
        if lcs_length == 0 or len(generated_tokens) == 0 or len(reference_tokens) == 0:
            F1 = 0
        else:
            P = lcs_length / len(generated_tokens)
            R = lcs_length / len(reference_tokens)
            F1 = 2 * P * R / (P + R)
            assert 0 <= F1 <= 1, f"F1 score should be between 0 and 1, but got {F1} (P={P}, R={R}, LCS={lcs_length}, gen={len(generated_tokens)}, ref={len(reference_tokens)})"
        return_tuple += (F1,)
    if return_text:
        return_tuple += (generated_text, reference_text)
    return return_tuple if len(return_tuple) > 1 else return_tuple[0]

# %%

# def compare_texts(generated_text, label_text, tokenizer, k):
#     # Tokenize the texts
#     generated_tokens = tokenizer.encode(generated_text)
#     label_tokens = tokenizer.encode(label_text)

#     # Truncate the first k tokens
#     generated_tokens = generated_tokens[:k]
#     label_tokens = label_tokens[:k]

#     # Compute the longest common subsequence
#     lcs_length = longest_common_subsequence_length(generated_tokens, label_tokens)

#     return lcs_length
