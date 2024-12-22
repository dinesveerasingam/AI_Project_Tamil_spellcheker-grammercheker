import json
import re
import os
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class GrammarChecker:
    def __init__(self, svo_rules_path, pronoun_rules_path):
        self.svo_rules = self.load_rules(svo_rules_path)
        self.pronoun_rules = self.load_rules(pronoun_rules_path)

    @staticmethod
    def load_rules(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Failed to load rules from {filepath}: File not found.")
            return {}
        except json.JSONDecodeError as e:
            print(f"Failed to load rules from {filepath}: {e}")
            return {}

    def correct_svo_order(self, sentence):
        for rule in self.svo_rules.get("svo_order_rules", []):
            pattern = re.compile(rule["pattern"], re.IGNORECASE)
            match = pattern.search(sentence)
            if match:
                corrected = pattern.sub(rule["correction"], sentence)
                return corrected, "SVO Order Error"
        return sentence, None

    def correct_pronoun_usage(self, sentence):
        original_sentence = sentence
        for incorrect_pronoun, correct_pronoun in self.pronoun_rules.get("pronoun_mapping", {}).items():
            pattern = re.compile(r'\b' + re.escape(incorrect_pronoun) + r'\b', re.IGNORECASE)
            sentence = pattern.sub(correct_pronoun, sentence)
        if sentence != original_sentence:
            return sentence, "Pronoun Usage Error"
        return sentence, None

    def check_grammar(self, sentence):
        corrected_sentence, error = self.correct_svo_order(sentence)
        if error:
            corrected_sentence, pronoun_error = self.correct_pronoun_usage(corrected_sentence)
            if pronoun_error:
                return corrected_sentence, f"{error}, {pronoun_error}"
        return corrected_sentence, "No errors detected."

def save_results_to_file(test_sentences, reference_sentences, grammar_checker, filename="grammar_checker_results.txt"):
    cc = SmoothingFunction()
    with open(filename, "w", encoding="utf-8") as file:
        for i, sentence in enumerate(test_sentences):
            corrected_sentence, error_type = grammar_checker.check_grammar(sentence)
            reference_sentence = reference_sentences[i]
            bleu_score = sentence_bleu([reference_sentence.split()], corrected_sentence.split(), smoothing_function=cc.method4)
            
            result_text = f"Sentence {i+1}:\nOriginal: {sentence}\nCorrected: {corrected_sentence}\nReference: {reference_sentence}\nError Type: {error_type}\nBLEU Score: {bleu_score:.4f}\n\n"
            file.write(result_text)

    # Open the file in Notepad
    os.system(f"notepad.exe {filename}")

# Initialize the GrammarChecker with paths to your rule files
grammar_checker = GrammarChecker("svo_order_rules.json", "pronoun_rules.json")

# Example test and reference sentences to evaluate
test_sentences = [
    "அவர் மகளிடம் பரிசு கொடுத்தாள்.",
    "அவள் கதையை கேட்டான்.",
    "அவனை அழைத்தாள் தோழன்.",
    "அவர் கைகளை தூக்கியாள்.",
    "அவனை பார்த்தான் சிறுவர்கள்.",
    "அவர் கதையை கூறினாள்.",
    "அவள் மகனை அழைத்தான்.",
    "மாணவர்களை கற்றார் ஆசிரியர் பாடத்தை.",
    "பூனைக்கு கொடுத்தான் சிறுவன் பாலை.",
    "தோட்டத்தில் விளையாடினான் குழந்தைகள்.",
    "புத்தகத்தை வாசித்தாள் அவள் நூலகத்தில்.",
    "காய்களை வாங்கினான் அவர் சந்தையில்."
]

# Expected reference corrections and identified error types
reference_sentences = [
   "அவர் மகளிடம் பரிசு கொடுத்தார்.",
    "அவள் கதையை கேட்டாள்.",
    "அவளை அழைத்தான் தோழன்.", 
    "அவர் கைகளை தூக்கியார்.", 
    "அவளை பார்த்தார்கள் சிறுவர்கள்.", 
    "அவர் கதையை கூறினார்.", 
    "அவள் மகனை அழைத்தாள்.", 
    "ஆசிரியர் மாணவர்களுக்கு பாடத்தை கற்றார்.", 
    "சிறுவன் பூனைக்கு பாலை கொடுத்தான்.", 
    "குழந்தைகள் தோட்டத்தில் விளையாடினார்கள்.", 
    "அவள் நூலகத்தில் புத்தகத்தை வாசித்தாள்.", 
    "அவர் சந்தையில் காய்களை வாங்கினார்.", 
]

# Save the results to a file and open it in Notepad
save_results_to_file(test_sentences, reference_sentences, grammar_checker)
