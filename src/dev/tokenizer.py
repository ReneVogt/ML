from collections import defaultdict

def bytePairEncoding(text, mergeSteps = 0, debug = False):
    chars = list(set(text))
    vocabulary = {s:i for i,s in enumerate(chars)}
    decoder = {i:s for i,s in enumerate(chars)}
    tokens = [vocabulary[c] for c in text]
    separators = set([vocabulary[c] for c in vocabulary if not c.isalnum()])
    if mergeSteps == 0:
        return decoder, tokens
    if mergeSteps < 0:
        mergeSteps = 5 * len(decoder)

    tokenPairOccurences = defaultdict(int)
    for i in range(len(tokens)-1):
        if not tokens[i] in separators and not tokens[i+1] in separators:
            tokenPairOccurences[tokens[i],tokens[i+1]] += 1

    for step in range(mergeSteps):
        mergeTokenPair = max(tokenPairOccurences, key=tokenPairOccurences.get)
        occurences = tokenPairOccurences[mergeTokenPair]
        del tokenPairOccurences[mergeTokenPair]
        if debug:
            print(f'Merge {step}: pair {mergeTokenPair} occurs {occurences} in {len(tokens)} (ratio: {(occurences/len(tokens)):.4f}), vocab: {len(vocabulary)}.')
        mergedString = decoder[mergeTokenPair[0]] + decoder[mergeTokenPair[1]]
        mergedToken = len(vocabulary)
        vocabulary[mergedString] = mergedToken
        decoder[mergedToken] = mergedString

        index = 0
        while index < len(tokens) - 1:
            if (tokens[index],tokens[index+1]) != mergeTokenPair:
                index += 1
                continue

            if index > 0 and not tokens[index-1] in separators:
                leadingPair = tokens[index-1],mergeTokenPair[0]
                tokenPairOccurences[leadingPair] -= 1
                if tokenPairOccurences[leadingPair] == 0:
                    del tokenPairOccurences[leadingPair]
                tokenPairOccurences[tokens[index-1], mergedToken] += 1
            if index < len(tokens) - 2 and not tokens[index+2] in separators:
                trailingPair = mergeTokenPair[1],tokens[index+2]
                tokenPairOccurences[trailingPair] -= 1
                if tokenPairOccurences[trailingPair] == 0:
                    del tokenPairOccurences[trailingPair]
                tokenPairOccurences[mergedToken, tokens[index+2]] += 1

            tokens[index] = mergedToken
            del tokens[index+1]
            index += 1

    return decoder, tokens
                
            