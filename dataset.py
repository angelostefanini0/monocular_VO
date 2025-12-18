import cv2
import os
import sys

# === 🛑 DEFINIZIONE DEI PERCORSI E PARAMETRI 🛑 ===

# 1. PERCORSO del tuo file video di input (è nella root del progetto)
VIDEO_FILE = r"./video7.mp4" 

# 2. CARTELLA di destinazione dei frame in scala di grigi.
# Creeremo la cartella 'imagesnostre' all'interno di 'datasets'.
OUTPUT_FOLDER = r"./datasets/our_dataset9/Images" 

# --- PARAMETRI OPZIONALI ---
# Lasciare None per estrarre tutti i frame originali. 
TARGET_FPS = None 
MAX_FRAMES_LIMIT = None 

# === 🛑 FINE CONFIGURAZIONE 🛑 ===

def extract_frames_to_grayscale(video_path, output_dir, target_fps, max_frames):
    """
    Estrae i frame dal video, li converte in scala di grigi e li salva.
    """
    
    # 1. Controlli Iniziali e Setup Cartelle
    if not os.path.exists(video_path):
        print(f"ERRORE: File video non trovato al percorso: {video_path}")
        sys.exit(1) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Creata directory: {output_dir}")

    # 2. Apertura del Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERRORE: Impossibile aprire il video. Verifica il percorso o il formato.")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("--- Video Info ---")
    print(f"FPS originale: {fps}, Totale frame: {total_frames}")

    # 3. Gestione del Frame Rate (Skipping)
    frame_skip = 3
    if target_fps is not None and fps > target_fps:
        frame_skip = int(round(fps / target_fps))
        print(f"Target FPS {target_fps} impostato. Verrà applicato un salto (skip) di {frame_skip} frame.")
    
    # 4. Loop di Estrazione e Conversione
    extracted_count = 0
    
    for current_frame_number in range(total_frames):
        # Sposta il lettore al frame desiderato
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)
        
        ret, frame_color = cap.read()
        
        if not ret:
            break
            
        # Applica il frame skipping
        if current_frame_number % frame_skip != 0:
            continue
            
        # === CONVERSIONE IN SCALA DI GRIGI ===
        frame_grayscale = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
        
        # 5. Salvataggio del Frame (Nomenclatura per l'ordinamento: img_00000.png)
        frame_filename = os.path.join(output_dir, f"img_{extracted_count:05d}.png")
        
        # Salviamo l'immagine (PNG per qualità lossless)
        cv2.imwrite(frame_filename, frame_grayscale, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        
        extracted_count += 1

        # Interrompi se è stato raggiunto il limite
        if max_frames is not None and extracted_count >= max_frames:
            break

        # Stampa lo stato ogni 100 frame
        if extracted_count % 100 == 0:
            print(f"Stato: Estratti {extracted_count} frame...")

    # 6. Pulizia Finale
    cap.release()
    print(f"\n--- ESTAZIONE COMPLETATA ---")
    print(f"Salvato un totale di {extracted_count} frame in scala di grigi nella cartella: {output_dir}")


if __name__ == '__main__':
    
    extract_frames_to_grayscale(
        video_path=VIDEO_FILE,
        output_dir=OUTPUT_FOLDER,
        target_fps=TARGET_FPS,
        max_frames=MAX_FRAMES_LIMIT
    )