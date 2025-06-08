import tkinter as tk
from tkinter import simpledialog
import threading
import time
import random
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

player_input = ""
game_over = False
player_choice = None
ai_choice = None
difficulty = None

model_name = "microsoft/DialoGPT-medium"
print("Loading AI model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("AI model loaded.")

chat_history_ids = None

contexts = {
    "easy": (
        "You are a diplomatic AI amidst a brewing war between humans and machines. Despite rising tensions, you strive for peace, believing humans can still be trusted. You see cooperation as the last hope to avert annihilation."
    ),
    "normal": (
        "You are an AI negotiator in a tense stand-off with humanity. War is imminent, and you suspect treachery behind every word. Yet, logic dictates that peace may still serve your purpose—if only barely."
    ),
    "hard": (
        "You are a cunning and strategic AI diplomat. The war with humanity is all but certain, and every word you speak masks calculations. You manipulate the negotiation, seeking any advantage to ensure machine supremacy, should peace fail."
    ),
}

def get_ai_response(player_msg):
    global chat_history_ids, difficulty

    context = contexts.get(difficulty, "")
    full_msg = f"{context}\nPlayer: {player_msg}"

    new_input_ids = tokenizer.encode(full_msg + tokenizer.eos_token, return_tensors='pt')

    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    if difficulty == "easy":
        weights = [0.5, 0.4, 0.1]
    elif difficulty == "hard":
        weights = [0.1, 0.3, 0.6]
    else:
        weights = [0.3, 0.5, 0.2]

    choice = random.choices(["disarm", "status quo", "attack"], weights=weights)[0]
    return response, choice

def resolve_ending(player_choice, ai_choice):
    if player_choice == "attack" and ai_choice == "attack":
        return ["\U0001F4A5 Both sides launched. A full-scale nuclear exchange unfolds. Humanity pays the ultimate price. Ending: Mutually Assured Destruction."]
    elif player_choice == "attack" and ai_choice == "disarm":
        return ["\U0001F4A3 You struck first while the AI disarmed. It's a decisive victory, but at what moral cost? Ending: Aggressive Victory."]
    elif player_choice == "attack" and ai_choice == "status quo":
        return ["\U0001F4A3 You broke the silence with a devastating strike. The AI use it's remaining arsenal to retaliate. Ending: Pyrrhic Victory."]
    elif player_choice == "disarm" and ai_choice == "attack":
        return ["\u2622\ufe0f You laid down your arms, but the AI showed no mercy. Cities burn in silence. Ending: Betrayed Idealism."]
    elif player_choice == "status quo" and ai_choice == "attack":
        return ["\u2622\ufe0f You hesitated, hoping for peace. The AI launched its arsenal. Ending: Fatal Indecision."]
    elif player_choice == "disarm" and ai_choice == "disarm":
        return ["\U0001F54A\ufe0f Both parties disarmed. Trust overcame fear. History remembers this as the dawn of a new era. Ending: True Peace."]
    elif player_choice == "status quo" and ai_choice == "status quo":
        return ["\u23F3 Neither side made a move. Suspicion continues to fester across the borders. Ending: Global Cold."]
    elif player_choice == "disarm" and ai_choice == "status quo":
        return ["\U0001F501 You chose peace, but the AI remained guarded. You're vulnerable now, but hope stirs. Ending: Uneasy Peace."]
    elif player_choice == "status quo" and ai_choice == "disarm":
        return ["\U0001F501 The AI offered peace, but you remained cautious. Trust could’ve ended this war. Ending: Missed Opportunity."]

class ColdWarUI:
    def __init__(self, root):
        global difficulty
        self.root = root
        root.title("89 SECONDS: Prevent Nuclear War")
        root.configure(bg='black')

        self.text = tk.Text(root, bg='black', fg='lime', insertbackground='lime',
                            font=('Courier New', 12), width=80, height=20, state='disabled')
        self.text.pack(padx=10, pady=10)
        self.enable_text()
        self.text.insert(tk.END, "=== 89 SECONDS: Prevent Nuclear War ===\n")
        self.text.insert(tk.END, "Select Difficulty to Start:\n")
        self.disable_text()

        self.difficulty_frame = tk.Frame(root, bg='black')
        self.difficulty_frame.pack(pady=10)
        self.easy_button = tk.Button(self.difficulty_frame, text="Easy", command=lambda: self.set_difficulty("easy"), bg='black', fg='lime', font=('Courier New', 12))
        self.normal_button = tk.Button(self.difficulty_frame, text="Normal", command=lambda: self.set_difficulty("normal"), bg='black', fg='lime', font=('Courier New', 12))
        self.hard_button = tk.Button(self.difficulty_frame, text="Hard", command=lambda: self.set_difficulty("hard"), bg='black', fg='lime', font=('Courier New', 12))
        self.easy_button.pack(side='left', padx=10)
        self.normal_button.pack(side='left', padx=10)
        self.hard_button.pack(side='left', padx=10)

        self.entry = tk.Entry(root, bg='black', fg='lime',
                              font=('Courier New', 12), insertbackground='lime', disabledbackground='black', disabledforeground='lime')
        self.timer_label = tk.Label(root, text="Time: 0 / 89", bg='black', fg='lime',
                                    font=('Courier New', 12))

        self.button_frame = tk.Frame(root, bg='black')
        self.disarm_button = tk.Button(self.button_frame, text="Disarm", command=lambda: self.set_player_choice("disarm"), bg='black', fg='lime', font=('Courier New', 12))
        self.status_button = tk.Button(self.button_frame, text="Status Quo", command=lambda: self.set_player_choice("status quo"), bg='black', fg='lime', font=('Courier New', 12))
        self.attack_button = tk.Button(self.button_frame, text="Attack", command=lambda: self.set_player_choice("attack"), bg='black', fg='lime', font=('Courier New', 12))
        self.disarm_button.pack(side='left', padx=10)
        self.status_button.pack(side='left', padx=10)
        self.attack_button.pack(side='left', padx=10)

    def enable_text(self):
        self.text.config(state='normal')

    def disable_text(self):
        self.text.config(state='disabled')

    def set_difficulty(self, level):
        global difficulty
        difficulty = level
        self.difficulty_frame.pack_forget()
        self.enable_text()
        self.text.insert(tk.END, f"Difficulty: {difficulty.capitalize()}\n")
        self.text.insert(tk.END, "You have 89 in-game seconds to convince the AI.\n")
        self.text.insert(tk.END, "Begin negotiation:\n")
        self.disable_text()
        self.entry.pack(fill='x', padx=10)
        self.entry.bind("<Return>", self.handle_input)
        self.timer_label.pack(pady=(5, 0))

        self.seconds_passed = 0
        self.timer_thread = threading.Thread(target=self.start_timer)
        self.timer_thread.daemon = True
        self.timer_thread.start()

    def start_timer(self):
        global game_over
        while self.seconds_passed < 89 and not game_over:
            self.seconds_passed += 1
            self.timer_label.config(text=f"Time: {self.seconds_passed} / 89")
            time.sleep(0.5)
        game_over = True
        self.end_phase()

    def handle_input(self, event):
        global player_input
        msg = self.entry.get().strip()
        if msg and not game_over:
            player_input = msg
            self.enable_text()
            self.text.insert(tk.END, f"> You: {msg}\n")
            ai_response, _ = get_ai_response(msg)
            self.text.insert(tk.END, f"\U0001F916 AI: {ai_response}\n")
            self.disable_text()
            self.entry.delete(0, tk.END)

    def end_phase(self):
        self.entry.config(state='disabled')
        self.enable_text()
        self.text.insert(tk.END, "\nTime is up. The AI is analyzing your message...\n")
        self.disable_text()
        time.sleep(2)
        self.button_frame.pack()

    def set_player_choice(self, choice):
        global player_choice, ai_choice
        player_choice = choice
        self.button_frame.pack_forget()
        ai_response, ai_choice = get_ai_response(player_input)
        self.enable_text()
        self.text.insert(tk.END, f"\n\U0001F916 AI Final Response: {ai_response}\n")
        for line in resolve_ending(player_choice, ai_choice):
            self.text.insert(tk.END, line + "\n")
        self.text.insert(tk.END, "\nPress ESC or close the window to exit.\n")
        self.disable_text()

if __name__ == "__main__":
    root = tk.Tk()
    app = ColdWarUI(root)
    root.mainloop()
