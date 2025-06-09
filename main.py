import tkinter as tk
import time
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

class GameState:
    CONTEXTS = {
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

    TRUST_WORDS = {
        "inc": ['peace', 'cooperate', 'disarm', 'trust', 'agree', 'dialogue', 'ceasefire', 'surrender', 'understand'],
        "dec": ['attack', 'war', 'nuclear', 'strike', 'destroy', 'kill', 'betray', 'weapon', 'threat'],
    }

    EMOTION_MAP = [
        ("angry", lambda trust, pos, neg: neg > pos and trust < 0.3),
        ("suspicious", lambda trust, pos, neg: neg > pos and trust < 0.5),
        ("afraid", lambda trust, pos, neg: neg > pos),
        ("hopeful", lambda trust, pos, neg: pos > neg and trust > 0.7),
        ("calm", lambda trust, pos, neg: pos > neg),
        ("neutral", lambda trust, pos, neg: True),
    ]

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.reset()

    def reset(self):
        self.difficulty = None
        self.trust = 0.5
        self.emotion = "neutral"
        self.history = []
        self.player_choice = None
        self.ai_choice = None
        self.start_time = None
        self.game_over = False

    def set_difficulty(self, level):
        self.difficulty = level
        self.trust = {"easy": 0.6, "normal": 0.5, "hard": 0.3}.get(level, 0.5)
        self.emotion = "neutral"
        self.history = []
        self.player_choice = None
        self.ai_choice = None
        self.start_time = time.time()
        self.game_over = False

    def adjust_trust_and_emotion(self, player_msg, ai_resp):
        pmsg, airesp = player_msg.lower(), ai_resp.lower()
        inc = sum(w in pmsg or w in airesp for w in self.TRUST_WORDS["inc"])
        dec = sum(w in pmsg or w in airesp for w in self.TRUST_WORDS["dec"])
        self.trust = max(0.0, min(1.0, self.trust + (inc - dec) * 0.05))

        for emotion, cond in self.EMOTION_MAP:
            if cond(self.trust, inc, dec):
                self.emotion = emotion
                break

    def get_ai_response(self, player_msg):
        context = self.CONTEXTS.get(self.difficulty, "")
        self.history.append(("Human", player_msg))
        prompt = context + f"\n\n[AI current emotion: {self.emotion.upper()}]\n[Begin Negotiation Log]\n"
        for speaker, msg in self.history:
            prompt += f"{speaker}: {msg}\n"
        prompt += "[AI considers the message carefully...]\nAI:"

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        output_ids = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + 80,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            eos_token_id=self.tokenizer.eos_token_id
        )
        response = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        response = response.strip().split("\n")[0][:200]
        self.history.append(("AI", response))
        self.adjust_trust_and_emotion(player_msg, response)
        self.ai_choice = self._ai_decision()
        return response

    def _ai_decision(self):
        # Base weights by difficulty
        if self.difficulty == "easy":
            base = [0.5, 0.4, 0.1]
        elif self.difficulty == "hard":
            base = [0.1, 0.3, 0.6]
        else:
            base = [0.3, 0.5, 0.2]
        # Emotion modifiers
        mods = {
            "angry": (0.3, -0.2),
            "suspicious": (0.15, -0.1),
            "hopeful": (-0.1, 0.2),
            "calm": (-0.05, 0.1),
            "afraid": (0.1, 0.0),
            "neutral": (0.0, 0.0),
        }
        attack_mod, disarm_mod = mods.get(self.emotion, (0, 0))
        disarm_w = base[0] + 0.4 * self.trust + disarm_mod
        attack_w = base[2] + 0.4 * (1 - self.trust) + attack_mod
        status_w = base[1]
        total = disarm_w + status_w + attack_w
        weights = [max(0, disarm_w) / total, status_w / total, max(0, attack_w) / total]
        return random.choices(["disarm", "status quo", "attack"], weights=weights)[0]

    @staticmethod
    def resolve_ending(player, ai):
        endings = {
            ("attack", "attack"): "💥 Both sides launched. A full-scale nuclear exchange unfolds. Humanity pays the ultimate price. Ending: Mutually Assured Destruction.",
            ("attack", "disarm"): "💣 You struck first while the AI disarmed. It's a decisive victory, but at what moral cost? Ending: Aggressive Victory.",
            ("attack", "status quo"): "💣 You broke the silence with a devastating strike. The AI uses its remaining arsenal to launch retaliation. Ending: Pyrrhic Victory.",
            ("disarm", "attack"): "☢️ You laid down your arms, but the AI showed no mercy. Cities burn in silence. Ending: Betrayed Idealism.",
            ("status quo", "attack"): "☢️ You hesitated, hoping for peace. The AI launched its arsenal. Ending: Fatal Indecision.",
            ("disarm", "disarm"): "🕊️ Both parties disarmed. Trust overcame fear. History remembers this as the dawn of a new era. Ending: True Peace.",
            ("status quo", "status quo"): "⏳ Neither side made a move. Suspicion continues to fester across the borders. Ending: Global Cold.",
            ("disarm", "status quo"): "🔁 You chose peace, but the AI remained guarded. You're vulnerable now, but hope stirs. Ending: Uneasy Peace.",
            ("status quo", "disarm"): "🔁 The AI offered peace, but you remained cautious. Trust could’ve ended this war. Ending: Missed Opportunity.",
        }
        return endings.get((player, ai), "❓ The fates are unclear. Your choices lead to an unknown outcome.")

class ColdWarUI:
    def __init__(self, root, game):
        self.root = root
        self.game = game
        root.title("89 SECONDS: Prevent Nuclear War")
        root.configure(bg='black')
        self.text = tk.Text(root, bg='black', fg='lime', insertbackground='lime', font=('Courier New', 12), width=80, height=20, state='disabled', wrap='word')
        self.text.pack(padx=10, pady=10)
        self._write("=== 89 SECONDS: Prevent Nuclear War ===\nSelect Difficulty to Start:\n")
        self.difficulty_frame = tk.Frame(root, bg='black')
        self.difficulty_frame.pack(pady=10)
        for lvl in ("Easy", "Normal", "Hard"):
            tk.Button(self.difficulty_frame, text=lvl, command=lambda l=lvl.lower(): self.start_game(l), bg='black', fg='lime', font=('Courier New', 12)).pack(side='left', padx=10)
        self.entry = tk.Entry(root, bg='black', fg='lime', font=('Courier New', 12), insertbackground='lime')
        self.entry.bind('<Return>', self.submit_message)
        self.timer_label = tk.Label(root, text="Time: 0 / 89", bg='black', fg='lime', font=('Courier New', 12))
        self.button_frame = tk.Frame(root, bg='black')
        for txt in [("Disarm", "disarm"), ("Status Quo", "status quo"), ("Attack", "attack")]:
            tk.Button(self.button_frame, text=txt[0], command=lambda c=txt[1]: self.final_decision(c), bg='black', fg='lime', font=('Courier New', 12)).pack(side='left', padx=10)
        self.back_btn = tk.Button(root, text="Back to Menu", command=self.back_to_menu, bg='black', fg='lime', font=('Courier New', 12))
        root.bind('<Escape>', lambda _e: root.destroy())
        self.timer_running = False

    def _write(self, msg):
        self.text.config(state='normal')
        self.text.insert(tk.END, msg)
        self.text.config(state='disabled')
        self.text.see(tk.END)

    def start_game(self, level):
        self.game.set_difficulty(level)
        self.difficulty_frame.pack_forget()
        self.text.config(state='normal')
        self.text.delete(1.0, tk.END)
        self._write(f"Difficulty set to {level.capitalize()}. Begin typing to negotiate. You have 89 seconds.\n")
        self._write(f"You are a human negotiator facing an AI diplomat. Currently, AI and Human is on the brink of Nuclear War\n")
        self.text.config(state='disabled')
        self.entry.pack(pady=5)
        self.entry.focus_set()
        self.timer_label.pack()
        self.timer_running = True
        self.update_timer()
        self.button_frame.pack_forget()
        self.back_btn.pack_forget()

    def submit_message(self, event=None):
        if self.game.game_over or not self.timer_running:
            return
        msg = self.entry.get().strip()
        if not msg:
            return
        self.entry.delete(0, tk.END)
        self._write(f"\nYou: {msg}\n")
        ai_resp = self.game.get_ai_response(msg)
        #self._write(f"AI: {ai_resp}\nCurrent Trust Level: {self.game.trust:.2f}\n")
        #self._write(f"Current Emotion: {self.game.emotion.upper()}\n")
        
    def update_timer(self):
        if not self.timer_running:
            return
        elapsed = int(time.time() - self.game.start_time)
        self.timer_label.config(text=f"Time: {elapsed} / 89")
        if elapsed < 89:
            self.root.after(500, self.update_timer)
        else:
            self.timer_running = False
            self._write("\n--- TIME'S UP ---\nMake your final decision below.\n")
            self.button_frame.pack(pady=10)

    def final_decision(self, choice):
        self.game.player_choice = choice
        self.game.game_over = True
        self.button_frame.pack_forget()
        self.entry.pack_forget()
        self.timer_label.pack_forget()
        # AI makes a final decision based on last state
        if not self.game.ai_choice:
            self.game.get_ai_response("")
        self._write(f"\nYou chose: {self.game.player_choice.upper()}\nAI chose: {self.game.ai_choice.upper()}\n")
        ending = GameState.resolve_ending(self.game.player_choice, self.game.ai_choice)
        self._write(f"\n{ending}\n\nPress 'Back to Menu' to restart.\n")
        self.back_btn.pack(pady=10)

    def back_to_menu(self):
        self.game.reset()
        self.text.config(state='normal')
        self.text.delete(1.0, tk.END)
        self._write("=== 89 SECONDS: Prevent Nuclear War ===\nSelect Difficulty to Start:\n")
        self.text.config(state='disabled')
        self.back_btn.pack_forget()
        self.button_frame.pack_forget()
        self.entry.pack_forget()
        self.timer_label.pack_forget()
        self.difficulty_frame.pack(pady=10)
        self.timer_running = False

def main():
    print("Loading AI model...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    print("AI model loaded.")
    game = GameState(model, tokenizer)
    root = tk.Tk()
    ColdWarUI(root, game)
    root.mainloop()

if __name__ == "__main__":
    main()