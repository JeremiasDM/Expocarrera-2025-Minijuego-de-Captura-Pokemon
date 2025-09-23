import os
import cv2
import mediapipe as mp
import random
import numpy as np
import time

# Constantes de resolución
FRAME_W, FRAME_H = 1366, 768

# Paleta de colores (BGR)
COLOR_TEXTO_PRINCIPAL = (84, 53, 55)    # #373554 (gris-azulado oscuro)
COLOR_BOTONES = (239, 216, 160)         # #A0D8EF (azul cielo pastel)
COLOR_BOTONES_SELECCION = (219, 196, 140) # Sombra para los botones no seleccionados
COLOR_DESTACADOS = (207, 177, 178)      # #B2B1CF (violeta grisáceo)
COLOR_ATENCION = (207, 177, 178)        # #B2B1CF (violeta grisáceo)
COLOR_FONDO = (0, 20, 64)           # #ECE9F7 (lila muy claro)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pokemon_folder = os.path.join(BASE_DIR, "Pokemon")
scores_path = os.path.join(BASE_DIR, "scores.txt")
achievements = {
    10: "Novato Aventura",
    20: "Maestro Pokebola",
    30: "Leyenda Pokeball",
    40: "Super Entrenador",
    50: "¡Campeón de la Liga!",
    60: "Maestro Pokemon"
}

# Verificación de carpetas y archivos
if not os.path.exists(pokemon_folder):
    print(f"Error: La carpeta '{pokemon_folder}' no existe. Por favor, crea la carpeta y agrega tus imágenes PNG de Pokémon y Pokebola.")
    exit()

pokemons = [f for f in os.listdir(pokemon_folder) if f.lower().endswith(".png") and "ball" not in f.lower()]
pokeballs = [f for f in os.listdir(pokemon_folder) if "ball" in f.lower() and f.lower().endswith(".png")]

if not pokemons or not pokeballs:
    print("No hay imágenes de Pokemon o Pokeball en la carpeta Pokemon.")
    exit()

masterball_name = None
for ball in pokeballs:
    if "masterball" in ball.lower():
        masterball_name = ball
        break

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Variables de juego
score = 0
last_achievement_score = 0
vidas = 3
confeti_frames = 0
confeti_coords = []
achievement_unlocked_message = None
achievement_display_start_time = 0

# Variables de Pokebola
pokeball_active = False
pokeball_x, pokeball_y = 0, 0
pokeball_vx, pokeball_vy = 0, 0
pokemon_size = 120
pokeball_size = 60
pokeball_speed = 40

# Variables de Pokémon
current_pokemon_path = os.path.join(pokemon_folder, random.choice(pokemons))
current_pokemon = cv2.imread(current_pokemon_path, cv2.IMREAD_UNCHANGED)
current_pokeball = None
pokemon_x = random.randint(50, FRAME_W - 50 - pokemon_size)
pokemon_y = random.randint(50, FRAME_H - 50 - pokemon_size)
dx, dy = random.choice([-3, 3]), random.choice([-2, 2])

def load_png(path, default_size=120):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return np.zeros((default_size, default_size, 4), dtype=np.uint8)
    return img

def overlay_png(bg, fg, x, y, size):
    """Superpone una imagen PNG con transparencia sobre un fondo."""
    if fg is None:
        return bg
    fg_resized = cv2.resize(fg, (size, size))
    h_fg, w_fg, _ = fg_resized.shape
    h_bg, w_bg, _ = bg.shape

    # Calcular las coordenadas de la región de interés (ROI)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_fg), min(h_bg, y + h_fg)
    if x2 <= x1 or y2 <= y1:
        return bg
    
    # Recortar las imágenes si es necesario
    fg_roi = fg_resized[y1 - y:y2 - y, x1 - x:x2 - x]
    bg_roi = bg[y1:y2, x1:x2]

    alpha_fg = fg_roi[:, :, 3] / 255.0
    alpha_fg = np.stack([alpha_fg] * 3, axis=-1)
    fg_rgb = fg_roi[:, :, :3]
    
    composite = (alpha_fg * fg_rgb + (1 - alpha_fg) * bg_roi).astype(np.uint8)
    bg[y1:y2, x1:x2] = composite
    return bg

def draw_text_centered(frame, text, y, font_scale, color, thickness, font=cv2.FONT_HERSHEY_TRIPLEX):
    """Dibuja texto centrado horizontalmente."""
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (FRAME_W - text_width) // 2
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

def draw_rounded_rectangle(image, start_point, end_point, color, thickness=cv2.FILLED, corner_radius=15):
    """Dibuja un rectángulo con esquinas redondeadas."""
    x1, y1 = start_point
    x2, y2 = end_point
    
    cv2.rectangle(image, (x1 + corner_radius, y1), (x2 - corner_radius, y2), color, thickness)
    cv2.rectangle(image, (x1, y1 + corner_radius), (x2, y2 - corner_radius), color, thickness)
    
    cv2.circle(image, (x1 + corner_radius, y1 + corner_radius), corner_radius, color, thickness)
    cv2.circle(image, (x2 - corner_radius, y1 + corner_radius), corner_radius, color, thickness)
    cv2.circle(image, (x1 + corner_radius, y2 - corner_radius), corner_radius, color, thickness)
    cv2.circle(image, (x2 - corner_radius, y2 - corner_radius), corner_radius, color, thickness)

def mostrar_menu():
    opciones = ["JUGAR", "INSTRUCCIONES", "PUNTUACIONES"]
    seleccion = -1

    # Coordenadas de las opciones (x1, y1, x2, y2)
    x_btn, w_btn = (FRAME_W - 500) // 2, 500
    h_btn = 80
    coords_opciones = []
    for i in range(len(opciones)):
        y_btn = 300 + i * 120
        coords_opciones.append((x_btn, y_btn, x_btn + w_btn, y_btn + h_btn))

    def mouse_callback(event, x, y, flags, param):
        nonlocal seleccion
        if event == cv2.EVENT_LBUTTONDOWN:
            for idx, (x1, y1, x2, y2) in enumerate(coords_opciones):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    seleccion = idx

    cv2.namedWindow("Minijuego Pokemon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Minijuego Pokemon", FRAME_W, FRAME_H)
    cv2.setMouseCallback("Minijuego Pokemon", mouse_callback)

    while True:
        menu_frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
        # Fondo y título con degradado
        cv2.rectangle(menu_frame, (0, 0), (FRAME_W, FRAME_H), COLOR_FONDO, -1)
        draw_text_centered(menu_frame, "Minijuego Pokemon", 120, 3, COLOR_ATENCION, 7)
        
        # Dibujar botones
        for i, opcion in enumerate(opciones):
            x1, y1, x2, y2 = coords_opciones[i]
            color = COLOR_BOTONES if seleccion == i else COLOR_BOTONES_SELECCION
            draw_rounded_rectangle(menu_frame, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
            draw_text_centered(menu_frame, opcion, y1 + 55, 1.5, COLOR_TEXTO_PRINCIPAL, 3)

        draw_text_centered(menu_frame, "Haz clic en una opcion para seleccionar", FRAME_H - 50, 1.2, COLOR_TEXTO_PRINCIPAL, 2, font=cv2.FONT_HERSHEY_PLAIN)
        
        cv2.imshow("Minijuego Pokemon", menu_frame)
        key = cv2.waitKey(20)
        if seleccion != -1:
            cv2.destroyWindow("Minijuego Pokemon")
            return seleccion

def mostrar_instrucciones():
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    instrucciones = [
        "INSTRUCCIONES:",
        "- Mueve tu mano para lanzar la Pokebola al Pokémon.",
        "- Atrapa el Pokémon para sumar un punto.",
        "- Si fallas, pierdes una vida.",
        "- El juego termina al perder todas las vidas.",
        "- Presiona ESC para volver al menú principal."
    ]
    
    cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), COLOR_FONDO, -1)
    draw_text_centered(frame, "INSTRUCCIONES", 120, 2.5, COLOR_ATENCION, 5)

    y_start = 220
    for i, linea in enumerate(instrucciones):
        draw_text_centered(frame, linea, y_start + i * 60, 1.2, COLOR_TEXTO_PRINCIPAL, 2, font=cv2.FONT_HERSHEY_SIMPLEX)
    
    draw_text_centered(frame, "Presiona ESC para volver", FRAME_H - 50, 1, COLOR_TEXTO_PRINCIPAL, 2, font=cv2.FONT_HERSHEY_PLAIN)

    cv2.imshow("Instrucciones", frame)
    while True:
        key = cv2.waitKey(20)
        if key == 27:
            break
    cv2.destroyWindow("Instrucciones")

def mostrar_puntuaciones():
    puntuaciones = []
    if os.path.exists(scores_path):
        with open(scores_path, "r") as f:
            for linea in f:
                partes = linea.strip().split()
                if len(partes) == 2 and partes[1].isdigit():
                    puntuaciones.append((partes[0], int(partes[1])))
    puntuaciones.sort(key=lambda x: x[1], reverse=True)

    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), COLOR_FONDO, -1)
    draw_text_centered(frame, "PUNTUACIONES", 120, 2.5, COLOR_ATENCION, 5)
    
    board_x, board_y, board_w, board_h = (FRAME_W - 600) // 2, 200, 600, 400
    cv2.rectangle(frame, (board_x, board_y), (board_x + board_w, board_y + board_h), COLOR_TEXTO_PRINCIPAL, -1)
    draw_rounded_rectangle(frame, (board_x, board_y), (board_x + board_w, board_y + board_h), COLOR_DESTACADOS, 2)
    
    draw_text_centered(frame, "TOP 10", board_y + 40, 1, COLOR_BOTONES, 2)
    
    for i, (nombre, puntaje) in enumerate(puntuaciones[:10]):
        y = board_y + 80 + i * 30
        color = COLOR_DESTACADOS if i == 0 else COLOR_TEXTO_PRINCIPAL
        text_line = f"{i+1}. {nombre} - {puntaje}"
        draw_text_centered(frame, text_line, y, 0.8, color, 2)

    draw_text_centered(frame, "Presiona ESC para volver", FRAME_H - 50, 1, COLOR_TEXTO_PRINCIPAL, 2, font=cv2.FONT_HERSHEY_PLAIN)
    
    cv2.imshow("Puntuaciones", frame)
    while True:
        key = cv2.waitKey(20)
        if key == 27:
            break
    cv2.destroyWindow("Puntuaciones")

def draw_confetti(frame, cantidad=50):
    """Genera coordenadas de confeti."""
    h, w, _ = frame.shape
    coords = []
    for _ in range(cantidad):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        color = tuple([random.randint(0, 255) for _ in range(3)])
        radius = random.randint(2, 6)
        coords.append((x, y, color, radius))
    return coords

def show_confetti(frame, coords):
    """Dibuja confeti en el fotograma."""
    for x, y, color, radius in coords:
        cv2.circle(frame, (x, y), radius, color, -1)

def jugar():
    global score, vidas, confeti_frames, confeti_coords
    global current_pokemon, current_pokeball_path, current_pokeball
    global pokemon_x, pokemon_y, dx, dy, pokeball_active, pokeball_x, pokeball_y, pokeball_vx, pokeball_vy
    global last_achievement_score, achievement_unlocked_message, achievement_display_start_time

    score = 0
    vidas = 3
    last_achievement_score = 0
    achievement_unlocked_message = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cv2.namedWindow("Minijuego Pokemon", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Minijuego Pokemon", FRAME_W, FRAME_H)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        exit()

    # Cronómetro de 5 segundos
    countdown_start_time = time.time()
    while time.time() - countdown_start_time < 5:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        
        cv2.rectangle(frame, (0, 0), (FRAME_W, FRAME_H), COLOR_FONDO, -1)
        countdown_num = 5 - int(time.time() - countdown_start_time)
        text_to_show = f"Preparate en... {countdown_num}"
        draw_text_centered(frame, text_to_show, FRAME_H // 2, 2.5, COLOR_ATENCION, 5)
        cv2.imshow("Minijuego Pokemon", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Bucle principal del juego
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo leer el frame de la cámara.")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
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
            if not pokeball_active:
                for hand_landmarks in result.multi_hand_landmarks:
                    x = int(hand_landmarks.landmark[9].x * w)
                    y = int(hand_landmarks.landmark[9].y * h)
                    
                    selected_ball = masterball_name if random.random() < 0.05 and masterball_name else random.choice([b for b in pokeballs if b != masterball_name])
                    current_pokeball_path = os.path.join(pokemon_folder, selected_ball)
                    current_pokeball = load_png(current_pokeball_path, default_size=pokeball_size)
                    
                    pokeball_x, pokeball_y = x - pokeball_size // 2, y - pokeball_size // 2
                    pokeball_active = True
                    dir_x = pokemon_x + pokemon_size // 2 - (pokeball_x + pokeball_size // 2)
                    dir_y = pokemon_y + pokemon_size // 2 - (pokeball_y + pokeball_size // 2)
                    norm = max(1, (dir_x**2 + dir_y**2)**0.5)
                    pokeball_vx = int(pokeball_speed * dir_x / norm)
                    pokeball_vy = int(pokeball_speed * dir_y / norm)

        if pokeball_active and current_pokeball is not None:
            pokeball_x += pokeball_vx
            pokeball_y += pokeball_vy
            overlay_png(frame, current_pokeball, pokeball_x, pokeball_y, pokeball_size)
            
            pb_cx = pokeball_x + pokeball_size // 2
            pb_cy = pokeball_y + pokeball_size // 2
            pk_cx = pokemon_x + pokemon_size // 2
            pk_cy = pokemon_y + pokemon_size // 2
            dist = ((pb_cx - pk_cx)**2 + (pb_cy - pk_cy)**2)**0.5
            
            if dist < (pokemon_size // 2 + pokeball_size // 2):
                pokeball_active = False
                if current_pokeball_path == os.path.join(pokemon_folder, masterball_name):
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
        
        # Lógica de captura y fallo
        if atrapado:
            score += 1
            if score in achievements and score > last_achievement_score:
                achievement_unlocked_message = achievements[score]
                achievement_display_start_time = time.time()
                last_achievement_score = score
            
            confeti_frames = 30
            confeti_coords = draw_confetti(frame, cantidad=80)
            draw_text_centered(frame, "¡Pokemon Atrapado!", 100, 1.5, COLOR_ATENCION, 3)

            current_pokemon_path = os.path.join(pokemon_folder, random.choice(pokemons))
            current_pokemon = load_png(current_pokemon_path)
            pokemon_x = random.randint(50, w - 50 - pokemon_size)
            pokemon_y = random.randint(50, h - 50 - pokemon_size)
            dx, dy = random.choice([-3, 3]), random.choice([-2, 2])

        if fallo:
            vidas -= 1
            confeti_frames = 20
            confeti_coords = draw_confetti(frame, cantidad=40)
            draw_text_centered(frame, "¡Pokemon Escapó!", 100, 1.5, COLOR_DESTACADOS, 3)

        if confeti_frames > 0:
            show_confetti(frame, confeti_coords)
            confeti_frames -= 1
        
        # Mostrar logros
        if achievement_unlocked_message and (time.time() - achievement_display_start_time < 3):
            cv2.rectangle(frame, (w - 450, h - 100), (w - 10, h - 10), COLOR_TEXTO_PRINCIPAL, -1)
            cv2.putText(frame, "Logro Desbloqueado:", (w - 430, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_BOTONES, 2)
            cv2.putText(frame, achievement_unlocked_message, (w - 430, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_DESTACADOS, 2)
        else:
            achievement_unlocked_message = None

        # Mostrar estadísticas del juego
        cv2.putText(frame, f"Atrapados: {score}", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_BOTONES, 3)
        cv2.putText(frame, f"Vidas: {vidas}", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, COLOR_DESTACADOS, 3)

        if vidas == 0:
            nombre = ""
            while len(nombre) < 3:
                temp_frame = frame.copy()
                draw_text_centered(temp_frame, "GAME OVER", FRAME_H // 2 - 100, 2, COLOR_DESTACADOS, 5)
                draw_text_centered(temp_frame, f"Tu puntuacion: {score}", FRAME_H // 2, 1.5, COLOR_BOTONES, 3)
                draw_text_centered(temp_frame, f"Ingrese sus iniciales: {nombre}", FRAME_H // 2 + 100, 1.5, COLOR_TEXTO_PRINCIPAL, 3)
                cv2.imshow("Minijuego Pokemon", temp_frame)
                key = cv2.waitKey(0)
                if 65 <= key <= 90 or 97 <= key <= 122:
                    letra = chr(key).upper()
                    nombre += letra
                elif key == 8 and len(nombre) > 0:
                    nombre = nombre[:-1]
            
            with open(scores_path, "a") as f:
                f.write(f"{nombre} {score}\n")
            
            print("Puntuacion guardada. Volviendo al menú principal...")
            break

        cv2.imshow("Minijuego Pokemon", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    while True:
        opcion = mostrar_menu()
        if opcion == 0:
            jugar()
        elif opcion == 1:
            mostrar_instrucciones()
        elif opcion == 2:
            mostrar_puntuaciones()
