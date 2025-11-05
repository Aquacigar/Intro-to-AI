import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re
from collections import Counter#polarity
import pickle
import os
#simple polarity based text classofn
class SentimentAnalyzer:
    def __init__(self):
        self.positive_words = set()
        self.negative_words = set()
        self.trained = False
        
    def preprocess(self, text):
      
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        return words
    
    def train(self, training_data):
        
        pos_words = []
        neg_words = []
        
        for text, label in training_data:
            words = self.preprocess(text)
            if label == 'positive':
                pos_words.extend(words)
            else:
                neg_words.extend(words)
        
       
        pos_counter = Counter(pos_words)
        neg_counter = Counter(neg_words)
        
       
        self.positive_words = set([word for word, _ in pos_counter.most_common(100)])
        self.negative_words = set([word for word, _ in neg_counter.most_common(100)])
        
        self.trained = True
        
    def predict(self, text):
        """Predict sentiment of text"""
        if not self.trained:
            return "neutral", 0.5
        
        words = self.preprocess(text)
        pos_score = sum(1 for word in words if word in self.positive_words)
        neg_score = sum(1 for word in words if word in self.negative_words)
        
        total = pos_score + neg_score
        if total == 0:
            return "neutral", 0.5
        
        confidence = max(pos_score, neg_score) / total
        
        if pos_score > neg_score:
            return "positive", confidence
        elif neg_score > pos_score:
            return "negative", confidence
        else:
            return "neutral", 0.5

class SentimentGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentiment Analyzer ML")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')
        
        self.analyzer = SentimentAnalyzer()
        self.setup_ui()
        self.train_default_model()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="Sentiment Analysis Machine Learning", 
                        font=('Arial', 18, 'bold'), bg='#f0f0f0', fg='#333')
        title.pack(pady=15)
        
        # Input frame
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        tk.Label(input_frame, text="Enter text to analyze:", 
                font=('Arial', 11), bg='#f0f0f0').pack(anchor='w')
        
        self.input_text = scrolledtext.ScrolledText(input_frame, height=8, 
                                                    font=('Arial', 10), 
                                                    wrap=tk.WORD)
        self.input_text.pack(fill='both', expand=True, pady=5)
        
        # Analyze 
        self.analyze_btn = tk.Button(self.root, text="Analyze Sentiment", 
                                     command=self.analyze_sentiment,
                                     font=('Arial', 12, 'bold'), 
                                     bg='#4CAF50', fg='white',
                                     cursor='hand2', padx=20, pady=10)
        self.analyze_btn.pack(pady=10)
        
        # Result 
        result_frame = tk.Frame(self.root, bg='#f0f0f0')
        result_frame.pack(pady=10, padx=20, fill='both')
        
        tk.Label(result_frame, text="Analysis Result:", 
                font=('Arial', 11, 'bold'), bg='#f0f0f0').pack(anchor='w')
        
        self.result_label = tk.Label(result_frame, text="", 
                                     font=('Arial', 14, 'bold'), 
                                     bg='#f0f0f0', pady=10)
        self.result_label.pack()
        
        self.confidence_label = tk.Label(result_frame, text="", 
                                        font=('Arial', 10), 
                                        bg='#f0f0f0')
        self.confidence_label.pack()
        
    def train_default_model(self):
        """Train with default dataset"""
        training_data = [
            ("it's amazing!", "positive"),
            ("fantastic", "positive"),
            ("Great experience", "positive"),
            ("Excellent quality", "positive"),
            ("I'm very happy", "positive"),
            ("Best thing ", "positive"),
            ("Absolutely brilliant and perfect", "positive"),
            ("good ", "positive"),
            ("This is terrible", "negative"),
            ("hate", "negative"),
            ("worst", "negative"),
            ("unhappy", "negative"),
            ("Poor quality", "negative"),
            ("This is bad and useless", "negative"),
            ("Horrible experience", "negative"),
            ("bad", "negative"),
        ]
        self.analyzer.train(training_data)
        self.training_data = training_data
        
    def analyze_sentiment(self):
        """Analyze the input text"""
        text = self.input_text.get("1.0", tk.END).strip()
        
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text to analyze")
            return
        
        sentiment, confidence = self.analyzer.predict(text)
        
        # Update result display
        color_map = {
            'positive': '#4CAF50',
            'negative': '#f44336',
            'neutral': '#FF9800'
        }
        
        self.result_label.config(text=f"Sentiment: {sentiment.upper()}", 
                                fg=color_map[sentiment])
        self.confidence_label.config(text=f"Confidence: {confidence:.1%}")
        
    def add_training_data(self):
        """Add new training example and retrain"""
        text = self.train_text.get().strip()
        
        if not text:
            messagebox.showwarning("Input Error", "Please enter training text")
            return
        
        sentiment = self.sentiment_var.get()
        self.training_data.append((text, sentiment))
        
        
        self.analyzer.train(self.training_data)
        
        messagebox.showinfo("Success", "Model retrained with new data!")
        self.train_text.delete(0, tk.END)

def main():
    root = tk.Tk()
    app = SentimentGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
