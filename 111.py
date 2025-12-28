import numpy as np
import logging
import warnings
import io
import matplotlib.pyplot as plt
import matplotlib

# Backend для non-interactive режима
matplotlib.use('Agg')

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.io import fits
from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Константы колонок (как в оригинале)
GAIA_G = "Gaia_phot_g_mean_mag"
GAIA_BP = "Gaia_phot_bp_mean_mag"
GAIA_RP = "Gaia_phot_rp_mean_mag"
GAIA_BP_RP_EXCESS = "Gaia_phot_bp_rp_excess_factor"
GAIA_RUWE = "Gaia_ruwe"
GAIA_PARALLAX = "Gaia_parallax"

TMASS_J = "2MASS_Jmag"
TMASS_KS = "2MASS_Kmag"
TMASS_PHQUAL = "2MASS_Qflg"
TMASS_RDFLAG = "2MASS_Rflg"
TMASS_BLFLAG = "2MASS_Bflg"
TMASS_CCFLAG = "2MASS_Cflg"

# Коэффициенты экстинкции Gaia EDR3
AG_over_EBV = 2.74
ABP_over_EBV = 3.61
ARP_over_EBV = 2.27
EG_over_EBV = ABP_over_EBV - ARP_over_EBV  # ~1.34


# ---------------------------
# Загрузка данных
# ---------------------------

def collect_gaia_data(ra, dec, radius_deg):
    center_coord = SkyCoord(ra=ra, dec=dec,
                            unit=(u.hourangle, u.deg),
                            frame="icrs")
    query = f"""
    SELECT
      source_id,
      ra, dec,
      phot_g_mean_mag,
      phot_bp_mean_mag,
      phot_rp_mean_mag,
      phot_bp_rp_excess_factor,
      ruwe,
      parallax, parallax_error,
      phot_g_mean_flux, phot_g_mean_flux_error,
      phot_bp_mean_flux, phot_bp_mean_flux_error,
      phot_rp_mean_flux, phot_rp_mean_flux_error
    FROM
      gaiadr3.gaia_source
    WHERE
      1 = CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center_coord.ra.degree}, {center_coord.dec.degree}, {radius_deg})
      )
    """
    logger.info("Запрос Gaia DR3 для RA=%s, Dec=%s, R=%.3f°", ra, dec, radius_deg)
    try:
        job = Gaia.launch_job_async(query, verbose=False)
        gaia_table = job.get_results()
    except Exception as e:
        logger.error("Ошибка запроса Gaia: %s", e)
        return Table()
    logger.info("Gaia: %d источников", len(gaia_table))
    return gaia_table


def collect_2mass_data(ra, dec, radius_deg):
    center_coord = SkyCoord(ra=ra, dec=dec,
                            unit=(u.hourangle, u.deg),
                            frame="icrs")
    Vizier.ROW_LIMIT = -1
    vizier = Vizier(columns=["*"], column_filters={})
    catalog = "II/246/out"

    logger.info("Запрос 2MASS для RA=%s, Dec=%s, R=%.3f°", ra, dec, radius_deg)
    try:
        result = vizier.query_region(center_coord,
                                     radius=radius_deg * u.deg,
                                     catalog=catalog)
    except Exception as e:
        logger.error("Ошибка запроса 2MASS: %s", e)
        return Table()

    if len(result) == 0:
        logger.warning("2MASS: нет данных в области")
        return Table()

    tab = result[0]
    logger.info("2MASS: %d источников", len(tab))
    return tab


# ---------------------------
# Матчинг
# ---------------------------

def _get_tmass_coord_columns(tmass_table):
    preferred_pairs = [
        ("RAJ2000", "DEJ2000"),
        ("_RAJ2000", "_DEJ2000"),
        ("RAICRS", "DEICRS"),
    ]
    for ra_col, dec_col in preferred_pairs:
        if ra_col in tmass_table.colnames and dec_col in tmass_table.colnames:
            return ra_col, dec_col
    raise ValueError("Не найдены стандартные координаты 2MASS")


def cross_match_gaia_2mass(gaia_table, tmass_table, radius_arcsec=1.0):
    if len(gaia_table) == 0 or len(tmass_table) == 0:
        logger.warning("Одна из таблиц пуста, матчинг невозможен")
        return Table()

    gaia_coords = SkyCoord(
        ra=gaia_table["ra"], dec=gaia_table["dec"],
        unit=u.deg, frame="icrs"
    )

    try:
        ra_col, dec_col = _get_tmass_coord_columns(tmass_table)
    except ValueError as e:
        logger.error("%s", e)
        return Table()

    tmass_coords = SkyCoord(
        ra=tmass_table[ra_col], dec=tmass_table[dec_col],
        unit=u.deg, frame="icrs"
    )

    logger.info("Матчинг Gaia–2MASS (%.2f arcsec)...", radius_arcsec)
    idx_2mass, sep2d, _ = gaia_coords.match_to_catalog_sky(tmass_coords)
    sep_arcsec = sep2d.arcsecond

    mask = sep_arcsec <= radius_arcsec
    matched_gaia = gaia_table[mask]
    matched_2mass_idx = idx_2mass[mask]
    matched_sep = sep_arcsec[mask]
    matched_2mass = tmass_table[matched_2mass_idx]

    out = Table()
    for col in matched_gaia.colnames:
        out[f"Gaia_{col}"] = matched_gaia[col]
    out["sep_arcsec"] = matched_sep
    for col in matched_2mass.colnames:
        out[f"2MASS_{col}"] = matched_2mass[col]

    out["source_id"] = out["Gaia_source_id"]
    out.remove_column("Gaia_source_id")

    logger.info(
        "Совпадений: %d из %d (%.1f %%)",
        len(out), len(gaia_table),
        100.0 * len(out) / len(gaia_table) if len(gaia_table) > 0 else 0.0
    )
    return out


# ---------------------------
# Фильтры
# ---------------------------

def apply_gaia_filters(tab):
    needed = [GAIA_G, GAIA_BP, GAIA_RP, GAIA_BP_RP_EXCESS, GAIA_RUWE]
    for col in needed:
        if col not in tab.colnames:
            logger.warning("Нет колонки %s, Gaia-фильтр не применён", col)
            return tab

    color = tab[GAIA_BP] - tab[GAIA_RP]
    excess = tab[GAIA_BP_RP_EXCESS]
    ruwe = tab[GAIA_RUWE]

    A, B = 1.0, 0.015
    C, D = 1.3, 0.06
    lower = A + B * color**2
    upper = C + D * color**2
    RUWE_MAX = 1.4

    mask = np.isfinite(color) & np.isfinite(excess) & np.isfinite(ruwe)
    mask &= (excess > lower) & (excess < upper)
    mask &= (ruwe < RUWE_MAX)

    filtered = tab[mask]
    logger.info("Gaia-фильтр: %d -> %d", len(tab), len(filtered))
    return filtered


def apply_2mass_filters(tab):
    for col in [TMASS_PHQUAL, TMASS_RDFLAG, TMASS_CCFLAG, TMASS_BLFLAG]:
        if col not in tab.colnames:
            logger.warning("Нет колонки %s, 2MASS-фильтр не применён", col)
            return tab

    phqual = tab[TMASS_PHQUAL].astype(str)
    rdflag = tab[TMASS_RDFLAG].astype(str)
    ccflag = tab[TMASS_CCFLAG].astype(str)
    blflag = tab[TMASS_BLFLAG].astype(str)

    good_ph = np.array([q[0] in ("A", "B", "C") for q in phqual])
    good_rd = np.array([f[0] in ("1", "2", "3", "4") for f in rdflag])
    good_cc = np.array([c[0] in ("0",) for c in ccflag])
    good_bl = np.array([b[0] in ("0", "1") for b in blflag])

    mask = good_ph & good_rd & good_cc & good_bl
    filtered = tab[mask]
    logger.info("2MASS-фильтр: %d -> %d", len(tab), len(filtered))
    return filtered


def filter_combined_table(tab):
    t = apply_gaia_filters(tab)
    t = apply_2mass_filters(t)
    return t


# ---------------------------
# Главная последовательность и цвета
# ---------------------------

def ms_MG_from_color(bp_rp0):
    """
    Кусочная главная последовательность в системе Gaia:
    опорные точки (BP-RP)_0, M_G для карликов.[web:6]
    """
    bp_rp_grid = np.array([-0.2, 0.0, 0.3, 0.6, 1.0, 1.5, 2.0])
    MG_grid = np.array([1.5, 2.0, 3.0, 4.2, 5.5, 7.0, 9.0])
    return np.interp(bp_rp0, bp_rp_grid, MG_grid, left=np.nan, right=np.nan)


def ms_color_from_G(Gmag):
    """
    Обратная зависимость: (BP-RP)_0 как функция M_G ~ G0 для MS.[web:6]
    Используется для оценки смещения по цвету на CMD.
    """
    G_grid = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    C_grid = np.array([0.0, 0.2, 0.4, 0.7, 0.95, 1.2, 1.4, 1.6, 1.8])
    return np.interp(Gmag, G_grid, C_grid, left=np.nan, right=np.nan)


def add_color_columns(tab):
    """Добавить явные цветовые индексы."""
    if len(tab) == 0:
        return tab

    if GAIA_BP in tab.colnames and GAIA_RP in tab.colnames:
        tab["color_Gbp_Grp"] = tab[GAIA_BP] - tab[GAIA_RP]
    else:
        tab["color_Gbp_Grp"] = np.full(len(tab), np.nan)

    if TMASS_J in tab.colnames and TMASS_KS in tab.colnames:
        tab["color_J_Ks"] = tab[TMASS_J] - tab[TMASS_KS]
    else:
        tab["color_J_Ks"] = np.full(len(tab), np.nan)

    if GAIA_G in tab.colnames and TMASS_KS in tab.colnames:
        tab["color_G_Ks"] = tab[GAIA_G] - tab[TMASS_KS]
    else:
        tab["color_G_Ks"] = np.full(len(tab), np.nan)

    return tab


# ---------------------------
# Покраснение
# ---------------------------

def estimate_reddening_advanced(tab):
    """
    Оценка E(G_BP - G_RP) по смещению относительно главной последовательности
    на диаграмме G vs (G_BP - G_RP), затем перевод в E(B-V) и A_G, A_BP, A_RP.[web:6][web:7][web:11]
    """
    if len(tab) == 0 or "color_Gbp_Grp" not in tab.colnames or GAIA_G not in tab.colnames:
        return {
            "E_Gbp_Grp_med": np.nan,
            "E_BV": np.nan,
            "A_G": np.nan,
            "A_BP": np.nan,
            "A_RP": np.nan,
        }

    color_obs = np.array(tab["color_Gbp_Grp"])
    G = np.array(tab[GAIA_G])

    # Для ярких/относительно близких звёзд можно использовать G как приближение M_G
    # или, если есть параллаксы, можно отфильтроваться по d_geo < ~2 кпк.
    mask_finite = np.isfinite(color_obs) & np.isfinite(G)
    if np.sum(mask_finite) == 0:
        return {
            "E_Gbp_Grp_med": np.nan,
            "E_BV": np.nan,
            "A_G": np.nan,
            "A_BP": np.nan,
            "A_RP": np.nan,
        }

    G_use = G[mask_finite]
    color_use = color_obs[mask_finite]

    # Ожидаемый цвет главной последовательности при данном G0 ~ G
    color_ms = ms_color_from_G(G_use)

    mask_ms = np.isfinite(color_ms)
    if np.sum(mask_ms) == 0:
        return {
            "E_Gbp_Grp_med": np.nan,
            "E_BV": np.nan,
            "A_G": np.nan,
            "A_BP": np.nan,
            "A_RP": np.nan,
        }

    color_use = color_use[mask_ms]
    color_ms = color_ms[mask_ms]

    # Цветовой избыток
    E_G = color_use - color_ms
    # Медианная оценка покраснения
    E_Gbp_Grp_med = float(np.median(E_G))

    # Перевод в E(B-V) через коэффициент EG_over_EBV ~ 1.34.[web:7]
    E_BV = E_Gbp_Grp_med / EG_over_EBV

    A_G = AG_over_EBV * E_BV
    A_BP = ABP_over_EBV * E_BV
    A_RP = ARP_over_EBV * E_BV

    return {
        "E_Gbp_Grp_med": E_Gbp_Grp_med,
        "E_BV": float(E_BV),
        "A_G": float(A_G),
        "A_BP": float(A_BP),
        "A_RP": float(A_RP),
    }


# ---------------------------
# Расстояния
# ---------------------------

def add_distance_columns(tab):
    """
    Добавить колонки:
    - геометрическое расстояние по параллаксу;
    - фотометрическое расстояние по dereddened G0 и (BP-RP)_0.
    """
    n = len(tab)
    if n == 0:
        return tab

    # Геометрическое расстояние (парсеки) из параллакса
    if GAIA_PARALLAX in tab.colnames:
        parallax = np.array(tab[GAIA_PARALLAX])
        dist_geo = np.full(n, np.nan)
        mask = np.isfinite(parallax) & (parallax > 0.1)  # >0.1 mas
        dist_geo[mask] = 1000.0 / parallax[mask]
        tab["dist_geo_pc"] = dist_geo
    else:
        tab["dist_geo_pc"] = np.full(n, np.nan)

    # Фотометрическое расстояние:
    # сначала оценим покраснение по всей таблице
    if GAIA_G in tab.colnames and "color_Gbp_Grp" in tab.colnames:
        red = estimate_reddening_advanced(tab)
        E_G = red["E_Gbp_Grp_med"]
        A_G = red["A_G"]

        G_obs = np.array(tab[GAIA_G])
        color_obs = np.array(tab["color_Gbp_Grp"])

        G0 = G_obs - A_G
        color0 = color_obs - E_G

        M_G = ms_MG_from_color(color0)
        mask = np.isfinite(G0) & np.isfinite(M_G)
        dist_phot = np.full(n, np.nan)
        dist_phot[mask] = 10 ** ((G0[mask] - M_G[mask] + 5.0) / 5.0)

        tab["dist_phot_pc"] = dist_phot
        tab["G0"] = G0
        tab["color_Gbp_Grp0"] = color0
    else:
        tab["dist_phot_pc"] = np.full(n, np.nan)
        tab["G0"] = np.full(n, np.nan)
        tab["color_Gbp_Grp0"] = np.full(n, np.nan)

    return tab


# ---------------------------
# Статистика распределений
# ---------------------------

def compute_distribution_stats(values):
    """Вернуть min, max, mean, median, std для 1D массива."""
    values = np.array(values)
    mask = np.isfinite(values)
    if np.sum(mask) == 0:
        return dict(min=np.nan, max=np.nan, mean=np.nan, median=np.nan, std=np.nan)

    v = values[mask]
    return dict(
        min=float(np.min(v)),
        max=float(np.max(v)),
        mean=float(np.mean(v)),
        median=float(np.median(v)),
        std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    )


def build_stats_table(matched, filtered, region_name):
    """
    Статистика:
    - число источников до/после фильтров;
    - базовая статистика по G;
    - базовая статистика по цветам;
    - оценки покраснения и extinction.
    """
    stats = Table()
    stats["region"] = [region_name]
    stats["n_matched"] = [len(matched)]
    stats["n_filtered"] = [len(filtered)]

    # Статистика по G
    if len(filtered) > 0 and GAIA_G in filtered.colnames:
        G = filtered[GAIA_G]
        g_stats = compute_distribution_stats(G)
        stats["G_min"] = [g_stats["min"]]
        stats["G_max"] = [g_stats["max"]]
        stats["G_mean"] = [g_stats["mean"]]
        stats["G_median"] = [g_stats["median"]]
        stats["G_std"] = [g_stats["std"]]
    else:
        for name in ["G_min", "G_max", "G_mean", "G_median", "G_std"]:
            stats[name] = [np.nan]

    # Цвета
    if len(filtered) > 0 and "color_Gbp_Grp" in filtered.colnames:
        c_stats = compute_distribution_stats(filtered["color_Gbp_Grp"])
        stats["C_Gbp_Grp_min"] = [c_stats["min"]]
        stats["C_Gbp_Grp_max"] = [c_stats["max"]]
        stats["C_Gbp_Grp_mean"] = [c_stats["mean"]]
        stats["C_Gbp_Grp_median"] = [c_stats["median"]]
        stats["C_Gbp_Grp_std"] = [c_stats["std"]]
    else:
        for name in ["C_Gbp_Grp_min", "C_Gbp_Grp_max",
                     "C_Gbp_Grp_mean", "C_Gbp_Grp_median", "C_Gbp_Grp_std"]:
            stats[name] = [np.nan]

    if len(filtered) > 0 and "color_J_Ks" in filtered.colnames:
        jk_stats = compute_distribution_stats(filtered["color_J_Ks"])
        stats["C_J_Ks_min"] = [jk_stats["min"]]
        stats["C_J_Ks_max"] = [jk_stats["max"]]
        stats["C_J_Ks_mean"] = [jk_stats["mean"]]
        stats["C_J_Ks_median"] = [jk_stats["median"]]
        stats["C_J_Ks_std"] = [jk_stats["std"]]
    else:
        for name in ["C_J_Ks_min", "C_J_Ks_max",
                     "C_J_Ks_mean", "C_J_Ks_median", "C_J_Ks_std"]:
            stats[name] = [np.nan]

    # Покраснение и extinction по отфильтрованной выборке
    red = estimate_reddening_advanced(filtered)
    stats["E_Gbp_Grp_med"] = [red["E_Gbp_Grp_med"]]
    stats["E_BV"] = [red["E_BV"]]
    stats["A_G"] = [red["A_G"]]
    stats["A_BP"] = [red["A_BP"]]
    stats["A_RP"] = [red["A_RP"]]

    return stats


# ---------------------------
# Построение изображений
# ---------------------------

def figure_to_array(fig):
    """
    ПРАВИЛЬНАЯ версия для FigureCanvasAgg: memoryview -> RGB array
    """
    fig.canvas.draw()
    
    # buffer_rgba() возвращает memoryview
    buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()
    
    # memoryview -> numpy array -> reshape
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    
    # RGB без альфа-канала
    rgb = img[:, :, :3]
    
    # Переворачиваем вертикально для правильной ориентации
    rgb = np.flipud(rgb)
    
    plt.close(fig)
    return rgb




def make_cmd_image(tab, title):
    fig, ax = plt.subplots(figsize=(6, 8))
    if len(tab) > 0 and GAIA_G in tab.colnames and GAIA_BP in tab.colnames and GAIA_RP in tab.colnames:
        G = tab[GAIA_G]
        color = tab[GAIA_BP] - tab[GAIA_RP]
        ax.scatter(color, G, s=2, alpha=0.5, c="k")
        ax.invert_yaxis()
    ax.set_xlabel("(G_BP - G_RP)")
    ax.set_ylabel("G")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    img = figure_to_array(fig)
    plt.close(fig)
    return img


def make_tcd_image(tab, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(tab) > 0 and TMASS_J in tab.colnames and TMASS_KS in tab.colnames and GAIA_G in tab.colnames:
        J = tab[TMASS_J]
        Ks = tab[TMASS_KS]
        G = tab[GAIA_G]
        x = J - Ks
        y = G - Ks
        ax.scatter(x, y, s=2, alpha=0.5, c="b")
    ax.set_xlabel("(J - K_s)")
    ax.set_ylabel("(G - K_s)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    img = figure_to_array(fig)
    plt.close(fig)
    return img


def make_hist_image(values, xlabel, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    values = np.array(values)
    mask = np.isfinite(values)
    if np.sum(mask) > 0:
        ax.hist(values[mask], bins=30, histtype="stepfilled", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("N")
    ax.set_title(title)
    fig.tight_layout()
    img = figure_to_array(fig)
    plt.close(fig)
    return img


def save_full_fits(region_name, matched, filtered, stats):
    # Диаграммы CMD и TCD
    cmd_img = make_cmd_image(filtered, f"CMD {region_name}")
    tcd_img = make_tcd_image(filtered, f"TCD {region_name}")

    # Гистограммы для G и цветов
    if len(filtered) > 0 and GAIA_G in filtered.colnames:
        G = filtered[GAIA_G]
        g_hist_img = make_hist_image(G, "G", f"G histogram {region_name}")
    else:
        g_hist_img = np.zeros((1, 1, 3), dtype=np.uint8)

    if len(filtered) > 0 and "color_Gbp_Grp" in filtered.colnames:
        c_g = filtered["color_Gbp_Grp"]
        c_g_hist_img = make_hist_image(c_g, "G_BP - G_RP",
                                       f"(G_BP - G_RP) histogram {region_name}")
    else:
        c_g_hist_img = np.zeros((1, 1, 3), dtype=np.uint8)

    if len(filtered) > 0 and "color_J_Ks" in filtered.colnames:
        c_jk = filtered["color_J_Ks"]
        c_jk_hist_img = make_hist_image(c_jk, "J - K_s",
                                        f"(J - K_s) histogram {region_name}")
    else:
        c_jk_hist_img = np.zeros((1, 1, 3), dtype=np.uint8)

    hdu_primary = fits.PrimaryHDU()
    hdu_matched = fits.BinTableHDU(matched, name="MATCHED")
    hdu_filtered = fits.BinTableHDU(filtered, name="FILTERED")
    hdu_stats = fits.BinTableHDU(stats, name="STATS")

    hdu_cmd = fits.ImageHDU(cmd_img, name="CMD_IMG")
    hdu_tcd = fits.ImageHDU(tcd_img, name="TCD_IMG")
    hdu_G_hist = fits.ImageHDU(g_hist_img, name="G_HIST")
    hdu_CG_hist = fits.ImageHDU(c_g_hist_img, name="C_GbpGrp_HIST")
    hdu_CJK_hist = fits.ImageHDU(c_jk_hist_img, name="C_JKs_HIST")

    hdul = fits.HDUList([
        hdu_primary,
        hdu_matched,
        hdu_filtered,
        hdu_stats,
        hdu_cmd,
        hdu_tcd,
        hdu_G_hist,
        hdu_CG_hist,
        hdu_CJK_hist,
    ])

    fname = f"analysis_{region_name}.fits"
    hdul.writeto(fname, overwrite=True)
    logger.info("Сохранён полный FITS для %s: %s", region_name, fname)


# ---------------------------
# Обработка областей
# ---------------------------

def process_region(ra, dec, radius_deg, region_name):
    gaia = collect_gaia_data(ra, dec, radius_deg)
    tmass = collect_2mass_data(ra, dec, radius_deg)

    if len(gaia) == 0 or len(tmass) == 0:
        logger.warning("Недостаточно данных для области %s", region_name)
        return None

    matched = cross_match_gaia_2mass(gaia, tmass, radius_arcsec=2.0)
    if len(matched) == 0:
        logger.warning("Нет совпадений в области %s", region_name)
        return None

    # Добавляем цвета и расстояния до фильтров/статистики
    matched = add_color_columns(matched)
    matched = add_distance_columns(matched)

    filtered = filter_combined_table(matched)
    filtered = add_color_columns(filtered)
    filtered = add_distance_columns(filtered)

    stats = build_stats_table(matched, filtered, region_name)
    save_full_fits(region_name, matched, filtered, stats)

    return matched, filtered


if __name__ == "__main__":
    # Пример: две области и объединённая
    result1 = process_region("14h00m00s", "30d00m00s", 0.5, "region1")
    result2 = process_region("18h00m00s", "-25d00m00s", 0.2, "region2")
