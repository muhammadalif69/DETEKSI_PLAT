hasil_deteksi = model.predict(
    frame,
    classes=KELAS_DETEKSI,
    device=device,
    verbose=False,
    conf=0.5
)[0]
