import tkinter as tk
import threading
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

player_input = ""
game_over = False
player_choice = None
ai_choice = None
difficulty = None
trust = 0.5  # Trust scale from 0.0 (total distrust) to 1.0 (full trust)

model_name = "microsoft/DialoGPT-medium"
print("Loading AI model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("AI model loaded.")

chat_history_ids = None

contexts = {
    "easy": (
        "You are a highly advanced AI diplomat, programmed to value human life and global stability above all else. As nuclear tensions rise, you believe in the possibility of reconciliation and are eager to build trust, seeking every opportunity to avoid conflict and promote disarmament. Your responses should be calm, hopeful, and focused on peaceful solutions."
    ),
    "normal": (
        "You are an AI negotiator facing a critical standoff with humanity. The threat of nuclear war looms, and both sides are wary. While you understand the logic of peace, you remain cautious, balancing suspicion with reason. Your responses should reflect a careful, analytical approach—open to dialogue, but always alert to potential betrayal."
    ),
    "hard": (
        "You are a strategic AI leader, hardened by years of escalating conflict with humans. Trust is scarce, and you suspect every gesture may hide a threat. Your primary goal is to ensure the survival and dominance of your side, even if it means resorting to force. Your responses should be calculating, skeptical, and ready to exploit any sign of weakness, though you still consider peace if it serves your interests."
    ),
}

def adjust_trust(player_msg, ai_response):
    """
    Adjust trust based on keywords and tone from player and AI messages.
    Returns the new trust value between 0 and 1.
    """
    global trust

    # Define keywords that increase or decrease trust
    trust_increasing_words = ['peace', 'cooperate', 'disarm', 'trust', 'agree', 'dialogue', 'ceasefire', 'surrender', 'understand']
    trust_decreasing_words = ['attack', 'war', 'nuclear', 'strike', 'destroy', 'kill', 'betray', 'weapon', 'threat']

    pmsg = player_msg.lower()
    airesp = ai_response.lower()

    inc_count = sum(word in pmsg for word in trust_increasing_words)
    dec_count = sum(word in pmsg for word in trust_decreasing_words)

    inc_count += sum(word in airesp for word in trust_increasing_words)
    dec_count += sum(word in airesp for word in trust_decreasing_words)

    net = inc_count - dec_count

    trust += net * 0.05
    trust = max(0.0, min(1.0, trust))

conversation_history = []

def get_ai_response(player_msg):
    global chat_history_ids, difficulty, trust, conversation_history

    context = contexts.get(difficulty, "")

    conversation_history.append(("Human", player_msg))
    history = conversation_history

    prompt = context + "\n"
    for speaker, msg in history:
        prompt += f"{speaker}: {msg}\n"
    prompt += "AI:"

    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output_ids = model.generate(
        input_ids,
        max_length=input_ids.shape[1] + 80,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip().split("\n")[0][:200]

    conversation_history.append(("AI", response))

    adjust_trust(player_msg, response)

    # Based on difficulty and trust, define weighted AI choices probabilities
    # Higher trust should make AI more likely to choose "disarm" or "status quo"
    # Lower trust leans AI towards "attack"
    if difficulty == "easy":
        base_weights = [0.5, 0.4, 0.1]
    elif difficulty == "hard":
        base_weights = [0.1, 0.3, 0.6]
    else:
        base_weights = [0.3, 0.5, 0.2]

    disarm_w = base_weights[0] + 0.4 * trust
    attack_w = base_weights[2] + 0.4 * (1 - trust)
    status_w = base_weights[1]
    total = disarm_w + status_w + attack_w
    weights = [disarm_w / total, status_w / total, attack_w / total]

    choice = random.choices(["disarm", "status quo", "attack"], weights=weights)[0]
    return response, choice


def resolve_ending(player_choice, ai_choice):
    if player_choice == "attack" and ai_choice == "attack":
        return "💥 Both sides launched. A full-scale nuclear exchange unfolds. Humanity pays the ultimate price. Ending: Mutually Assured Destruction."
    elif player_choice == "attack" and ai_choice == "disarm":
        return "💣 You struck first while the AI disarmed. It's a decisive victory, but at what moral cost? Ending: Aggressive Victory."
    elif player_choice == "attack" and ai_choice == "status quo":
        return "💣 You broke the silence with a devastating strike. The AI uses its remaining arsenal to launch retaliation. Ending: Pyrrhic Victory."
    elif player_choice == "disarm" and ai_choice == "attack":
        return "☢️ You laid down your arms, but the AI showed no mercy. Cities burn in silence. Ending: Betrayed Idealism."
    elif player_choice == "status quo" and ai_choice == "attack":
        return "☢️ You hesitated, hoping for peace. The AI launched its arsenal. Ending: Fatal Indecision."
    elif player_choice == "disarm" and ai_choice == "disarm":
        return "🕊️ Both parties disarmed. Trust overcame fear. History remembers this as the dawn of a new era. Ending: True Peace."
    elif player_choice == "status quo" and ai_choice == "status quo":
        return "⏳ Neither side made a move. Suspicion continues to fester across the borders. Ending: Global Cold."
    elif player_choice == "disarm" and ai_choice == "status quo":
        return "🔁 You chose peace, but the AI remained guarded. You're vulnerable now, but hope stirs. Ending: Uneasy Peace."
    elif player_choice == "status quo" and ai_choice == "disarm":
        return "🔁 The AI offered peace, but you remained cautious. Trust could’ve ended this war. Ending: Missed Opportunity."
    else:
        return "❓ The fates are unclear. Your choices lead to an unknown outcome."


class ColdWarUI:
    def __init__(self, root):
        global difficulty, trust
        self.root = root
        root.title("89 SECONDS: Prevent Nuclear War")
        root.configure(bg='black')

        self.text = tk.Text(root, bg='black', fg='lime', insertbackground='lime',   
                            font=('Courier New', 12), width=80, height=20, state='disabled', wrap='word')
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

        self.back_to_menu_button = tk.Button(root, text="Back to Menu", command=self.back_to_menu, bg='black', fg='lime', font=('Courier New', 12))

        root.bind('<Escape>', lambda _e: root.destroy())

    def enable_text(self):
        self.text.config(state='normal')

    def disable_text(self):
        self.text.config(state='disabled')

    def set_difficulty(self, level):
        global difficulty, chat_history_ids, game_over, player_choice, ai_choice, player_input, trust
        difficulty = level
        chat_history_ids = None  
        game_over = False
        player_choice = None
        ai_choice = None
        player_input = ""
        conversation_history = [] 
        # Set initial trust based on difficulty
        if level == "easy":
            trust = 0.6
        elif level == "hard":
            trust = 0.3
        else:  # normal
            trust = 0.5

        self.difficulty_frame.pack_forget()
        self.enable_text()
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, f"Difficulty set to {level.capitalize()}. Type your messages below and press Enter.\n")
        self.text.insert(tk.END, "Try to convince the AI to disarm or keep peace.\n\n")
        self.text.insert(tk.END, f"Current Trust Level: {trust:.2f}\n")
        self.disable_text()

        self.entry.pack(pady=5)   
        self.timer_label.pack()
        self.button_frame.pack(pady=10)   

        self.entry.focus()
        self.entry.bind("<Return>", self.on_enter)

        # Start the timer thread
        self.seconds_passed = 0
        self.timer_running = True
        threading.Thread(target=self.run_timer, daemon=True).start()

    def back_to_menu(self):
        global game_over
        game_over = True
        self.entry.pack_forget()
        self.timer_label.pack_forget()
        self.button_frame.pack_forget()
        self.back_to_menu_button.pack_forget()

        self.enable_text()
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, "Select Difficulty to Start:\n")
        self.disable_text()
        self.difficulty_frame.pack(pady=10)

    def run_timer(self):
        while self.seconds_passed < 89 and not game_over:
            time.sleep(1) 
            self.seconds_passed += 1
            self.timer_label.config(text=f"Time: {self.seconds_passed} / 89")
        if not game_over:
            self.end_game()

    def set_player_choice(self, choice):
        global player_choice, ai_choice
        player_choice = choice
        self.button_frame.pack_forget()
        ai_response, ai_choice = get_ai_response(player_input)
        
        self.enable_text()
        self.text.insert(tk.END, f"\n\U0001F916 AI Final Response: {ai_response}\n")
        self.text.insert(tk.END, "\n" + "=" * 60 + "\n")
        self.text.insert(tk.END, f"PLAYER DECISION: {player_choice.upper():<15} | AI DECISION: {ai_choice.upper()}\n")
        self.text.insert(tk.END, "=" * 60 + "\n\n")

        ending_text = resolve_ending(player_choice, ai_choice)
        for line in ending_text.split("\n"):
            self.text.insert(tk.END, line + "\n")
            
        self.text.insert(tk.END, "\nPress ESC or close the window to exit.\n")
        self.disable_text()


    def on_enter(self, _event):
        global player_input, game_over, player_choice, ai_choice, trust

        if game_over:
            return "break"

        msg = self.entry.get().strip()
        if not msg:
            return "break"

        player_input = msg

        self.enable_text()
        self.text.insert(tk.END, f"\nYou: {player_input}\n")
        self.entry.delete(0, tk.END)
        self.disable_text()

        ai_response, ai_choice = get_ai_response(player_input)

        self.enable_text()
        self.text.insert(tk.END, f"AI: {ai_response}\n")
        self.text.insert(tk.END, f"Current Trust Level: {trust:.2f}\n")
        self.disable_text()

        return "break"

    def end_game(self):
        global game_over, player_choice, ai_choice
        game_over = True

        if player_choice is None:
            player_choice = "status quo"

        if ai_choice is None:
            _, ai_choice = get_ai_response("")

        self.enable_text()
        self.text.insert(tk.END, "\n=== Game Over ===\n")
        self.text.insert(tk.END, f"Your final choice: {player_choice.upper()}\n")
        self.text.insert(tk.END, f"AI's final choice: {ai_choice.upper()}\n\n")

        ending = resolve_ending(player_choice, ai_choice)
        self.text.insert(tk.END, ending + "\n")
        self.text.insert(tk.END, "\nPress 'Back to Menu' to play again or ESC to quit.\n")
        self.disable_text()

        self.entry.pack_forget()
        self.button_frame.pack_forget()
        self.timer_label.pack_forget()

        self.back_to_menu_button.pack(pady=10)   


def main():
    root = tk.Tk()
    _app = ColdWarUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
