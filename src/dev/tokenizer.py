from collections import defaultdict

def tokenize(text, mergeSteps):
    chars = list(set(text))
    vocabulary = {s:i for i,s in enumerate(chars)}
    decoder = {i:s for i,s in enumerate(chars)}
    tokens = [vocabulary[c] for c in text]
    separators = set([vocabulary[c] for c in vocabulary if not c.isalnum()])

    for _ in range(mergeSteps):
        tokenPairOccurences = defaultdict(int)
        for i in range(len(tokens)-1):
            if not tokens[i] in separators and not tokens[i+1] in separators:
                tokenPairOccurences[tokens[i],tokens[i+1]] += 1
        
        mergeTokenPair = max(tokenPairOccurences, key=tokenPairOccurences.get)
        mergedString = decoder[mergeTokenPair[0]] + decoder[mergeTokenPair[1]]
        mergedToken = len(vocabulary)
        vocabulary[mergedString] = mergedToken
        decoder[mergedToken] = mergedString

        mergedTokens = []
        skip = False
        for i in range(len(tokens)):
            if skip:
                skip = False
                continue
            if i < len(tokens)-1 and (tokens[i],tokens[i+1]) == mergeTokenPair:
                mergedTokens.append(mergedToken)
                skip = True
            else:
                mergedTokens.append(tokens[i])
        tokens = mergedTokens

    return decoder, tokens
                
            