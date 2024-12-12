import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from AITester import load_model, process_image, predict_image


class Application:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Water vs Earth Classifier")
        self.root.geometry("800x600")  # Tamanho da janela
        self.root.config(bg="white")

        self.create_main_menu()

    def create_main_menu(self):
        """Cria o menu principal da aplicação."""
        # Limpar qualquer widget anterior
        for widget in self.root.winfo_children():
            widget.destroy()

        title = tk.Label(self.root, text="Menu Principal", font=("Arial", 24), bg="white")
        title.pack(pady=30)

        train_button = tk.Button(self.root, text="Treinar AI", command=self.train_model, width=20, height=2)
        train_button.pack(pady=10)

        test_button = tk.Button(self.root, text="Testar Imagem", command=self.test_model, width=20, height=2)
        test_button.pack(pady=10)

    def train_model(self):
        """Inicia o treinamento do modelo."""
        try:
            from AITrainer import train_ai_model
            train_ai_model()  # Chama a função de treinamento
            messagebox.showinfo("Sucesso", "Modelo treinado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar o modelo: {e}")

    def test_model(self):
        """Inicia o processo de teste da imagem."""
        self.create_test_interface()

    def create_test_interface(self):
        """Cria a interface para testar a imagem."""
        # Limpar a tela
        for widget in self.root.winfo_children():
            widget.destroy()

        title = tk.Label(self.root, text="Teste de Imagem", font=("Arial", 24), bg="white")
        title.pack(pady=30)

        # Botão para carregar a imagem
        test_button = tk.Button(self.root, text="Carregar Imagem", command=self.load_image, width=20, height=2)
        test_button.pack(pady=10)

        # Botão para voltar ao menu
        back_button = tk.Button(self.root, text="Voltar ao Menu", command=self.create_main_menu, width=20, height=2)
        back_button.pack(pady=10)

    def load_image(self):
        """Permite o usuário selecionar e testar a imagem."""
        model_path = "models/water_earth_model.pkl"
        try:
            model = load_model(model_path)
        except FileNotFoundError as e:
            messagebox.showerror("Erro", str(e))
            return

        image_path = filedialog.askopenfilename(
            title="Escolha uma imagem", filetypes=[("Image files", "*.jpg *.png")]
        )
        if not image_path:
            return

        try:
            # Processar e prever a imagem
            image_resized, image_pixels = process_image(image_path)
            predicted_mask = predict_image(image_pixels, model)

            # Mostrar o resultado
            self.show_result(image_resized, predicted_mask)

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao testar a imagem: {e}")

    def show_result(self, original, mask):
        """Exibe a imagem original e a máscara prevista lado a lado."""
        # Limpar a tela
        for widget in self.root.winfo_children():
            widget.destroy()

        # Criar um frame centralizado para o resultado
        result_frame = tk.Frame(self.root, bg="white")
        result_frame.pack(expand=True)

        # Converter as imagens para o formato PIL
        original_pil = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        mask_pil = Image.fromarray(mask)

        # Redimensionar as imagens
        original_img = ImageTk.PhotoImage(original_pil.resize((400, 400)))
        mask_img = ImageTk.PhotoImage(mask_pil.resize((400, 400)))

        # Exibir a imagem original
        original_label = tk.Label(result_frame, image=original_img)
        original_label.image = original_img
        original_label.grid(row=0, column=0, padx=10, pady=10)
        original_title = tk.Label(result_frame, text="Original", bg="white")
        original_title.grid(row=1, column=0)

        # Exibir a máscara prevista
        mask_label = tk.Label(result_frame, image=mask_img)
        mask_label.image = mask_img
        mask_label.grid(row=0, column=1, padx=10, pady=10)
        mask_title = tk.Label(result_frame, text="Máscara Prevista", bg="white")
        mask_title.grid(row=1, column=1)

        # Botão para voltar ao menu
        back_button = tk.Button(
            result_frame, text="Voltar ao Menu", command=self.create_main_menu, width=20, height=2
        )
        back_button.grid(row=2, column=0, columnspan=2, pady=20)


def main():
    root = tk.Tk()
    app = Application(root)
    root.mainloop()


if __name__ == "__main__":
    main()
