import os
import cv2
import mediapipe as mp
import random
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pokemon_folder = os.path.join(BASE_DIR, "Pokemon")

if not os.path.exists(pokemon_folder):
    print(f"Error: La carpeta '{pokemon_folder}' no existe. Crea la carpeta y agrega tus imágenes PNG de Pokémon y Pokeball.")
    exit()

pokemons = [f for f in os.listdir(pokemon_folder) if f.lower().endswith(".png") and "ball" not in f.lower()]
pokeballs = [f for f in os.listdir(pokemon_folder) if "ball" in f.lower() and f.lower().endswith(".png")]

if not pokeballs:
    print("No hay imágenes de Pokeball en la carpeta Pokemon.")
    exit()

masterball_name = None
for ball in pokeballs:
    if "masterball" in ball.lower():
        masterball_name = ball
        break

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

def load_png(path, default_size=120):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Error: No se pudo cargar {path}")
        return np.zeros((default_size, default_size, 4), dtype=np.uint8)
    return img

FRAME_W, FRAME_H = 1280, 720
pokemon_size = 120
pokeball_size = 60
pokeball_speed = 40

score = 0
vidas = 3
confeti_frames = 0
confeti_coords = []

pokeball_active = False
pokeball_x, pokeball_y = 0, 0

current_pokemon_path = os.path.join(pokemon_folder, random.choice(pokemons))
current_pokemon = load_png(current_pokemon_path)
current_pokeball_path = os.path.join(pokemon_folder, random.choice(pokeballs))
current_pokeball = load_png(current_pokeball_path, default_size=60)

pokemon_x = random.randint(50, FRAME_W - 50 - pokemon_size)
pokemon_y = random.randint(50, FRAME_H - 50 - pokemon_size)
dx, dy = random.choice([-3, 3]), random.choice([-2, 2])

def overlay_png(bg, fg, x, y, size):
    fg = cv2.resize(fg, (size, size))
    h_bg, w_bg = bg.shape[:2]
    x_end = min(x + size, w_bg)
    y_end = min(y + size, h_bg)
    x_start = max(x, 0)
    y_start = max(y, 0)
    fg_x_start = max(0, -x)
    fg_y_start = max(0, -y)
    fg_x_end = fg_x_start + (x_end - x_start)
    fg_y_end = fg_y_start + (y_end - y_start)
    if x_start < x_end and y_start < y_end:
        fg_crop = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]
        roi = bg[y_start:y_end, x_start:x_end]
        if fg_crop.shape[2] == 4:
            alpha_fg = fg_crop[:, :, 3] / 255.0
            alpha_fg = np.stack([alpha_fg]*3, axis=-1)
            fg_rgb = fg_crop[:, :, :3]
            np.copyto(roi, (alpha_fg * fg_rgb + (1 - alpha_fg) * roi).astype(np.uint8))
        else:
            np.copyto(roi, fg_crop)
    return bg

def draw_confetti(frame, cantidad=50):
    h, w, _ = frame.shape
    coords = []
    for _ in range(cantidad):
        x = random.randint(0, w-1)
        y = random.randint(0, h-1)
        color = tuple([random.randint(0,255) for _ in range(3)])
        radius = random.randint(2, 6)
        coords.append((x, y, color, radius))
    return coords

def show_confetti(frame, coords):
    for x, y, color, radius in coords:
        cv2.circle(frame, (x, y), radius, color, -1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

cv2.namedWindow("Minijuego Pokemon", cv2.WINDOW_NORMAL)  # Ventana ajustable

if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer el frame de la cámara.")
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (FRAME_W, FRAME_H))  # Puedes quitar esta línea si quieres que se adapte automáticamente

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    pokemon_x += dx
    pokemon_y += dy
    h, w, _ = frame.shape
    if pokemon_x < 0 or pokemon_x + pokemon_size > w:
        dx *= -1
        pokemon_x = max(0, min(pokemon_x, w - pokemon_size))
    if pokemon_y < 0 or pokemon_y + pokemon_size > h:
        dy *= -1
        pokemon_y = max(0, min(pokemon_y, h - pokemon_size))

    atrapado = False
    fallo = False

    if result.multi_hand_landmarks and vidas > 0:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if not pokeball_active:
                x = int(hand_landmarks.landmark[9].x * w)
                y = int(hand_landmarks.landmark[9].y * h)
                if masterball_name and random.random() < 0.01:
                    selected_ball = masterball_name
                else:
                    other_balls = [b for b in pokeballs if b != masterball_name] if masterball_name else pokeballs
                    selected_ball = random.choice(other_balls)
                current_pokeball_path = os.path.join(pokemon_folder, selected_ball)
                current_pokeball = load_png(current_pokeball_path, default_size=60)
                pokeball_x, pokeball_y = x - pokeball_size//2, y - pokeball_size//2
                pokeball_active = True
                dir_x = pokemon_x + pokemon_size//2 - (pokeball_x + pokeball_size//2)
                dir_y = pokemon_y + pokemon_size//2 - (pokeball_y + pokeball_size//2)
                norm = max(1, (dir_x**2 + dir_y**2)**0.5)
                pokeball_vx = int(pokeball_speed * dir_x / norm)
                pokeball_vy = int(pokeball_speed * dir_y / norm)

    if pokeball_active and current_pokeball is not None:
        pokeball_x += pokeball_vx
        pokeball_y += pokeball_vy
        overlay_png(frame, current_pokeball, pokeball_x, pokeball_y, pokeball_size)
        pb_cx = pokeball_x + pokeball_size//2
        pb_cy = pokeball_y + pokeball_size//2
        pk_cx = pokemon_x + pokemon_size//2
        pk_cy = pokemon_y + pokemon_size//2
        dist = ((pb_cx - pk_cx)**2 + (pb_cy - pk_cy)**2)**0.5
        if dist < (pokemon_size//2 + pokeball_size//2):
            pokeball_active = False
            if masterball_name and selected_ball == masterball_name:
                atrapado = True
            else:
                if random.random() > 0.33:
                    atrapado = True
                else:
                    fallo = True
        if (pokeball_x < 0 or pokeball_x > w - pokeball_size or
            pokeball_y < 0 or pokeball_y > h - pokeball_size):
            pokeball_active = False

    if current_pokemon is not None:
        overlay_png(frame, current_pokemon, pokemon_x, pokemon_y, pokemon_size)

    if atrapado:
        score += 1
        confeti_frames = 30
        confeti_coords = draw_confetti(frame, cantidad=80)
        cv2.putText(frame, "El Pokemon fue atrapado", (w//2-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 255, 0), 5)
        current_pokemon_path = os.path.join(pokemon_folder, random.choice(pokemons))
        current_pokemon = load_png(current_pokemon_path)
        pokemon_x = random.randint(50, w - 50 - pokemon_size)
        pokemon_y = random.randint(50, h - 50 - pokemon_size)
        dx, dy = random.choice([-3, 3]), random.choice([-2, 2])

    if fallo:
        vidas -= 1
        confeti_frames = 20
        confeti_coords = draw_confetti(frame, cantidad=40)
        cv2.putText(frame, "El Pokemon escapo", (w//2-200, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                    2, (0, 0, 255), 5)

    if confeti_frames > 0:
        show_confetti(frame, confeti_coords)
        confeti_frames -= 1

    cv2.putText(frame, f"Atrapados: {score}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 5)
    cv2.putText(frame, f"Vidas: {vidas}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)

    if vidas == 0:
        nombre = ""
        while len(nombre) < 3:
            temp_frame = frame.copy()
            cv2.putText(temp_frame, "GAME OVER", (w//2-200, h//2-50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)
            cv2.putText(temp_frame, f"Tu puntuacion: {score}", (w//2-200, h//2+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 5)
            cv2.putText(temp_frame, f"Ingrese sus iniciales: {nombre}", (w//2-200, h//2+150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)
            cv2.imshow("Minijuego Pokemon", temp_frame)
            key = cv2.waitKey(0)
            if 65 <= key <= 90 or 97 <= key <= 122:
                letra = chr(key).upper()
                nombre += letra
            elif key == 8 and len(nombre) > 0:
                nombre = nombre[:-1]
        with open(os.path.join(BASE_DIR, "scores.txt"), "a") as f:
            f.write(f"{nombre} {score}\n")
        print("Puntuacion guardada. Reiniciando juego...")
        score = 0
        vidas = 3
        confeti_frames = 0
        confeti_coords = []
        current_pokemon_path = os.path.join(pokemon_folder, random.choice(pokemons))
        current_pokemon = load_png(current_pokemon_path)
        pokemon_x = random.randint(50, w - 50 - pokemon_size)
        pokemon_y = random.randint(50, h - 50 - pokemon_size)
        dx, dy = random.choice([-3, 3]), random.choice([-2, 2])
        continue

    cv2.imshow("Minijuego Pokemon", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()