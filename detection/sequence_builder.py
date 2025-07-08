def build_sequences(faces, T=10):
    return [faces[i:i+T] for i in range(0, len(faces) - T + 1, T)]
