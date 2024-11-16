from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from cap_submission import *

def inspect_data(
    args,
    n_imgs: int = 2,
    ):
    # TODO:
    # Define your vision processor and language tokenizer
    # You do not need to submit this part, or the ``cap_main.py'' file. 
    # This is simply here for debugging purpose.
    tokenizer = GPT2Tokenizer.from_pretrained(args.decoder)
    processor = ViTImageProcessor.from_pretrained(args.encoder)

    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    special_tokens = {'bos_token': '<|beginoftext|>', 'pad_token': '<|pad|>'}
    tokenizer.add_special_tokens(special_tokens)

    dataset = FlickrDataset(args, tokenizer=tokenizer, processor=processor)
    indices = np.random.randint(0, len(dataset), size=(n_imgs, ))
    # Visualize with matplotlib
    for i, idx in enumerate(indices):
        encoding = dataset[idx]
        print(encoding["labels"])
        print(encoding["captions"])
        img_path = os.path.join(args.root_dir, "images", encoding["path"])
        img = np.array(Image.open(img_path))


        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(img, cmap="gray")
        print()

    plt.show()
    del dataset


def main():
    args = Args()
    try:
        import paramparse
        paramparse.process(args)
    except ImportError:
        print("WARNING: You have not installed paramparse. Please manually edit the arguments.")

    # Load the model and processors once, outside the loop
    model, processor, tokenizer = load_trained_model("./" + args.name)
    
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    print("Press 'q' to quit when image window is focused, or close the image window to select another image.")
    
    while True:
        # Get file path using tkinter
        file_path = filedialog.askopenfilename()
        
        if not file_path:  # If user cancels file selection
            print("No file selected. Exiting...")
            break
            
        print(f"Selected file: {file_path}")
        
        # Generate caption
        caption = inference(file_path, model, processor, tokenizer)
        print(f"Generated caption: {caption}")

        # Display image with caption
        fig = plt.figure(figsize=(10, 8))
        img = np.array(Image.open(file_path))
        plt.imshow(img)
        plt.axis('off')
        plt.title(caption, wrap=True)
        
        # Add keyboard event handler
        def on_key(event):
            if event.key == 'q':
                plt.close('all')
                return False
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show(block=True)  # Block until window is closed
        
        # Ask if user wants to continue
        user_input = input("Press Enter to select another image or 'q' to quit: ")
        if user_input.lower() == 'q':
            break
    
    print("Exiting program...")
    root.destroy()

if __name__ == "__main__":
    main()