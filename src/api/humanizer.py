import random
import re
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

from src.features.phrases import AI_PHRASES
from src.features.extractor import extract_features
from src.utils.cosine import compute_cosine_similarity
from src.api.ollama_client import polish_text

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt_tab', quiet=True)


def heuristic_rewrite(text: str) -> str:
    # 1. Remove AI phrases
    lower_text = text.lower()
    for phrase in AI_PHRASES:
        if phrase in lower_text:
            # Simple replacement logic: just remove the phrase and adjust spacing
            # For a more robust system, we would replace with neutral alternatives
            text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', text, flags=re.IGNORECASE)
            
    # Clean up multiple spaces and punctuation artifacts left by removal
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r' ,\s*', ', ', text)
    text = text.strip()

    # 2. Re-segment sentences (Burstiness manipulation)
    sentences = sent_tokenize(text)
    new_sentences = []
    i = 0
    while i < len(sentences):
        s = sentences[i].strip()
        # Randomly join two short sentences if they exist
        if i + 1 < len(sentences) and len(s.split()) < 10 and len(sentences[i+1].split()) < 10:
            if random.random() < 0.5:
                # Join with a conjunction or semicolon
                joiner = random.choice([" and ", " but ", "; "])
                new_s = s.rstrip('.!?') + joiner + sentences[i+1].lower().strip()
                new_sentences.append(new_s)
                i += 2
                continue
        
        # Randomly split a very long sentence (simplistic split on ' and ' or ', ')
        if len(s.split()) > 25 and random.random() < 0.5:
            split_points = [", and ", " and ", "; "]
            split_done = False
            for sp in split_points:
                if sp in s:
                    parts = s.split(sp, 1)
                    new_sentences.append(parts[0].strip() + ".")
                    new_sentences.append(parts[1].capitalize().strip())
                    split_done = True
                    break
            if split_done:
                i += 1
                continue

        new_sentences.append(s)
        i += 1

    text = " ".join(new_sentences)

    # 3. Synonym Swap (Entropy/Perplexity manipulation)
    words = word_tokenize(text)
    new_words = []
    for word in words:
        if word.isalpha() and len(word) > 4 and random.random() < 0.1:
            synonyms = []
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    syn_word = l.name().replace('_', ' ')
                    if syn_word.lower() != word.lower() and syn_word.isalpha():
                        synonyms.append(syn_word)
            if synonyms:
                # Pick a random synonym
                new_word = random.choice(synonyms)
                # Try to match capitalization
                if word.istitle():
                    new_word = new_word.capitalize()
                new_words.append(new_word)
            else:
                new_words.append(word)
        else:
            new_words.append(word)

    # Reconstruct text (very basic detokenization)
    reconstructed = " ".join(new_words)
    reconstructed = re.sub(r' ([.,!?])', r'\1', reconstructed)
    return reconstructed


def humanize(text: str, use_llm: bool = True, similarity_threshold: float = 0.85) -> dict:
    original_metrics = extract_features(text)
    
    # Stage 1: Heuristic Rewrite
    heuristic_text = heuristic_rewrite(text)
    post_heuristic_metrics = extract_features(heuristic_text)
    sim_heuristic = compute_cosine_similarity(text, heuristic_text)

    result = {
        "original_text": text,
        "humanized_text": heuristic_text,
        "original_metrics": original_metrics,
        "post_heuristic_metrics": post_heuristic_metrics,
        "similarity_before": sim_heuristic,
        "polish_failed": False,
        "used_llm": False
    }

    # Stage 2: LLM Polish (Optional)
    if use_llm:
        llm_text = polish_text(heuristic_text)
        if llm_text:
            sim_llm = compute_cosine_similarity(text, llm_text)
            if sim_llm >= similarity_threshold:
                post_llm_metrics = extract_features(llm_text)
                result["humanized_text"] = llm_text
                result["post_llm_metrics"] = post_llm_metrics
                result["similarity_after"] = sim_llm
                result["used_llm"] = True
            else:
                # Fallback to heuristic
                result["polish_failed"] = True
                result["similarity_after"] = sim_llm
        else:
            result["polish_failed"] = True
            
    return result
