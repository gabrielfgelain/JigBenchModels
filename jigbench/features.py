"""Feature table: one row per test, with the bus C/ESR labels and spectral features.

Run with `python -m jigbench.features`; writes outputs/features.csv and
outputs/features.json (dataset version and settings used).
"""

import json
import re

import numpy as np
import pandas as pd

from jigbench.dataset import REPO_ROOT, dataset_version, test_data_dir

FS = 100e3  # PZ4000 sampling rate (1 s window, 100 kS/s)

# Bench components
R_ARM = np.array([0.11, 0.58, 0.345, 0.45])  # resistors selected by the R code bits (ohm)
C_BUS_UF = [324, 490, 626, 739, 837, 922, 994, 1056]  # bus capacitance by number of arms (uF)

FILE_PATTERN = re.compile(
    r"(\d)Arms_R(\d{4})_(\d+)W_(\d+)RPM_(\d{4}-\d{2}-\d{2})_(\d{2})_(\d{2})-(\d{2})(_REF)?\.csv"
)

OUTPUT_DIR = REPO_ROOT / "outputs"


def bus_esr(arms: int, code: str) -> float:
    """Equivalent ESR of the DC bus (ohm) for a number of arms and R code."""
    if arms == 0:
        return 0.2
    on = np.array([c == "1" for c in code])
    s = 0.51 / arms + 1 / np.sum(1 / R_ARM[on])
    return 0.1 + 0.1 * s / (0.1 + s)


def spectrum(x: np.ndarray):
    """Single-sided amplitude spectrum with a Hann window."""
    x = x - x.mean()
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    return np.fft.rfftfreq(len(x), 1 / FS), np.abs(X) * 2 / w.sum()


def peak(freq, amp, f0, width=3):
    """Largest amplitude within f0 +/- width (Hz), and its frequency."""
    m = (freq > f0 - width) & (freq < f0 + width)
    i = np.argmax(amp[m])
    return amp[m][i], freq[m][i]


def test_features(df: pd.DataFrame, rpm: int) -> dict:
    """Features of one test (one CSV already read)."""
    vdc = df["U2_dcbus_V"].values
    vm = df["U1_inverter_V"].values
    im = df["I1_motor_phase_A"].values
    fe = rpm / 30  # electrical frequency, 2 pole pairs

    freq, A = spectrum(vdc)
    a120, _ = peak(freq, A, 120)
    a240, _ = peak(freq, A, 240)
    a360, _ = peak(freq, A, 360)
    a6f, _ = peak(freq, A, 6 * fe)
    asw, fsw = peak(freq, A, 4990, 60)
    band = (freq > 4500) & (freq < 5500)  # whole switching band, rms
    asw_band = np.sqrt(np.sum(A[band] ** 2) / 2)

    X = np.fft.rfft(vdc - vdc.mean())
    X[np.fft.rfftfreq(len(vdc), 1 / FS) < 2000] = 0  # ripple above 2 kHz, rms in time
    hf_rms = np.std(np.fft.irfft(X, len(vdc)))

    fi, Ai = spectrum(im)
    fv, Av = spectrum(vm)
    i1, _ = peak(fi, Ai, fe)
    v1, _ = peak(fv, Av, fe)

    # rms of both motor phases after a ~5 kHz moving average, and motor voltage rms
    k = np.ones(20) / 20
    im_rms = np.mean([
        np.sqrt(np.mean(np.convolve(df[c].values, k, "same") ** 2))
        for c in ["I1_motor_phase_A", "I2_motor_phase_B"]
    ])
    vm_rms = np.sqrt(np.mean(vm ** 2))

    return dict(
        Vdc=vdc.mean(), Vdc_pp=np.ptp(vdc), Vdc_rms_ripple=np.std(vdc), A120=a120, A240=a240, A360=a360, A6f=a6f,
        Asw=asw, fsw=fsw, Asw_band=asw_band, HF_rms=hf_rms, I1=i1, V1=v1,
        Im_rms=im_rms, Vm_rms=vm_rms,
    )


def build_table() -> pd.DataFrame:
    rows = []
    files = sorted(test_data_dir().rglob("*.csv"))
    for k, path in enumerate(files, 1):
        m = FILE_PATTERN.fullmatch(path.name)
        if not m:
            print("skipped:", path.name)
            continue
        arms, code, power, rpm, day, hh, mm, ss, ref = m.groups()
        arms, power, rpm = int(arms), int(power), int(rpm)
        df = pd.read_csv(path, sep=";", decimal=",")
        rows.append(dict(
            file=path.name,
            point=f"{power}W_{rpm}RPM",
            combo=f"{arms}Arms_R{code}",
            arms=arms, r_code=code, power_W=power, rpm=rpm,
            time=pd.Timestamp(f"{day} {hh}:{mm}:{ss}"),
            is_ref=ref is not None,
            C_uF=C_BUS_UF[arms],
            ESR_mOhm=1000 * bus_esr(arms, code),
            **test_features(df, rpm),
        ))
        print(f"\r{k}/{len(files)}", end="", flush=True)
    print()
    return pd.DataFrame(rows)


if __name__ == "__main__":
    version = dataset_version()
    table = build_table()
    OUTPUT_DIR.mkdir(exist_ok=True)
    table.to_csv(OUTPUT_DIR / "features.csv", index=False)
    meta = dict(**version, n_tests=len(table), fs_Hz=FS)
    (OUTPUT_DIR / "features.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {len(table)} tests to {OUTPUT_DIR / 'features.csv'}")
