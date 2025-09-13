import os
import csv
import math
import importlib.util
from collections import defaultdict, Counter

# Configuration - Update these paths as needed
SCAN_ROOT = os.getenv("SCAN_ROOT", r"D:\\")
PROJECT_ANCHOR = os.getenv("PROJECT_ANCHOR", r"D:\Caltrans Construction Project Sync")
GROUP_DEPTH = int(os.getenv("GROUP_DEPTH", "7"))

# Output paths - Update these as needed
REPORT_CSV = os.getenv("REPORT_CSV", r"C:\temp\D_Drive_Laser_Intensity_Report.csv")
DISCOVERY_CSV = os.getenv("DISCOVERY_CSV", r"C:\temp\D_Drive_Laser_Discovery.csv")

TARGET_EXTS = {".las", ".laz", ".e57"}
CHUNK_SIZE = 1_000_000
SAMPLE_LIMIT_PER_FILE = 5_000_000

PHOTOG_SW_HINTS = {"pix4d","metashape","agisoft","contextcapture","realitycapture","dronedeploy","openmvg","openmvs"}
LIDAR_SW_HINTS  = {"riegl","riscan","terrascan","leica","topcon","trimble","optech","zoller","faro","ouster","velodyne","phoenix","teledyne"}

KNOWN_DIMS = {
    "x","y","z","intensity","return_number","number_of_returns","scan_direction_flag","edge_of_flight_line",
    "classification","synthetic","key_point","withheld","overlap","scan_angle","scan_angle_rank",
    "user_data","point_source_id","gps_time","red","green","blue","nir"
}

def human_size(num):
    units = ["B","KB","MB","GB","TB","PB"]
    if num is None or num <= 0:
        return "0 B"
    i = int(math.floor(math.log(num, 1024)))
    p = math.pow(1024, i)
    s = round(num / p, 2)
    return f"{s} {units[i]}"

def win_long(p):
    if os.name != "nt":
        return p
    p = os.path.abspath(p)
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p

def strip_long(p):
    if p.startswith("\\\\?\\UNC\\"):
        return "\\" + p[7:]
    if p.startswith("\\\\?\\"):
        return p[4:]
    return p

def env_check():
    info = {"laspy_ok": False, "lazrs_ok": False, "laspy_version": None}
    try:
        import laspy
        info["laspy_ok"] = True
        info["laspy_version"] = getattr(laspy, "__version__", "unknown")
    except Exception:
        info["laspy_ok"] = False
    info["lazrs_ok"] = importlib.util.find_spec("lazrs") is not None
    return info

def scan_lidar_files(root_dir):
    long_root = win_long(root_dir)
    for dirpath, _, filenames in os.walk(long_root):
        for name in filenames:
            ext = os.path.splitext(name)[1].lower()
            if ext in TARGET_EXTS:
                yield strip_long(os.path.join(dirpath, name))

def project_key_for(path):
    parent = os.path.dirname(os.path.abspath(path))
    anchor = os.path.abspath(PROJECT_ANCHOR)
    try:
        common = os.path.commonpath([anchor, parent])
    except Exception:
        common = ""
    if common == anchor:
        rel = os.path.relpath(parent, anchor)
        parts = [] if rel in (".", "") else rel.split(os.sep)
        if not parts:
            return parent
        depth = max(1, int(GROUP_DEPTH))
        key_parts = parts[:depth]
        return os.path.join(anchor, *key_parts)
    return parent

def _xy_area_from_header(hdr):
    try:
        x_min = float(getattr(hdr, "x_min"))
        x_max = float(getattr(hdr, "x_max"))
        y_min = float(getattr(hdr, "y_min"))
        y_max = float(getattr(hdr, "y_max"))
        z_min = float(getattr(hdr, "z_min"))
        z_max = float(getattr(hdr, "z_max"))
    except Exception:
        try:
            mins = getattr(hdr, "mins", None) or getattr(hdr, "min", None)
            maxs = getattr(hdr, "maxs", None) or getattr(hdr, "max", None)
            x_min, y_min, z_min = float(mins[0]), float(mins[1]), float(mins[2])
            x_max, y_max, z_max = float(maxs[0]), float(maxs[1]), float(maxs[2])
        except Exception:
            return None, None, None, None, None, None, None
    dx = max(0.0, x_max - x_min)
    dy = max(0.0, y_max - y_min)
    area = dx * dy
    return x_min, x_max, y_min, y_max, z_min, z_max, (area if area > 0 else None)

def _deg_area_to_m2(dx_deg, dy_deg, lat_deg):
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*lat) + 1.175 * math.cos(4*lat) - 0.0023 * math.cos(6*lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3*lat) + 0.118 * math.cos(5*lat)
    return max(0.0, dx_deg) * m_per_deg_lon * max(0.0, dy_deg) * m_per_deg_lat

def _infer_unit_from_path(path_lower):
    tokens = {"utm":"m", "meter":"m", "metre":"m", "wgs84":"deg", "longlat":"deg", "latlon":"deg", "deg":"deg",
              "usft":"usft", "us_survey_foot":"usft", "survey_foot":"usft", "ft":"ft", "feet":"ft",
              "stateplane":"ft", "spc":"ft"}
    for k, v in tokens.items():
        if k in path_lower:
            if v == "m":
                return "meter", 1.0, "path_hint"
            if v == "ft":
                return "foot", 0.3048, "path_hint"
            if v == "usft":
                return "us_survey_foot", 0.3048006096012192, "path_hint"
            if v == "deg":
                return "degree", None, "path_hint"
    return None, None, None

def _infer_degrees_from_coords(x_min, x_max, y_min, y_max):
    if x_min is None or y_min is None or x_max is None or y_max is None:
        return False
    x_ok = -200.0 <= x_min <= 200.0 and -200.0 <= x_max <= 200.0
    y_ok = -100.0 <= y_min <= 100.0 and -100.0 <= y_max <= 100.0
    if not (x_ok and y_ok):
        return False
    dx = abs(x_max - x_min)
    dy = abs(y_max - y_min)
    return (0.00001 <= dx <= 10.0) and (0.00001 <= dy <= 10.0)

def _linear_unit_factor_to_m(reader, path_lower, x_min, x_max, y_min, y_max):
    unit_name = None
    factor = None
    source = None
    try:
        hdr = reader.header
        crs = hdr.parse_crs()
        if crs:
            try:
                names = [ax.unit_name.lower() for ax in crs.axis_info if getattr(ax, "unit_name", None)]
            except Exception:
                names = []
            unit_name_raw = names[0] if names else (crs.name or "").lower()
            if unit_name_raw:
                source = "crs_axis"
                if "metre" in unit_name_raw or "meter" in unit_name_raw or unit_name_raw == "m":
                    unit_name, factor = "meter", 1.0
                elif "us survey foot" in unit_name_raw:
                    unit_name, factor = "us_survey_foot", 0.3048006096012192
                elif "foot" in unit_name_raw or "feet" in unit_name_raw or unit_name_raw == "ft":
                    unit_name, factor = "foot", 0.3048
                elif "degree" in unit_name_raw:
                    unit_name, factor = "degree", None
    except Exception:
        pass

    if unit_name is None:
        u, f, s = _infer_unit_from_path(path_lower)
        if u:
            return u, f, s or "path_hint"

    if unit_name is None and _infer_degrees_from_coords(x_min, x_max, y_min, y_max):
        return "degree", None, "coord_infer"

    if unit_name is None and all(v is not None for v in [x_min, x_max, y_min, y_max]):
        dx = abs(x_max - x_min)
        dy = abs(y_max - y_min)
        if 1.0 <= max(dx, dy) <= 500000.0:
            return "meter", 1.0, "magnitude_infer"
        if max(dx, dy) > 500000.0:
            return "foot", 0.3048, "magnitude_infer"

    return unit_name, factor, source or "unknown"

def _classify_lidar_like(has_nonzero, has_int_dim, gens, syst):
    sw_text = " ".join(list(gens | syst))
    has_photo_hint = any(h in sw_text for h in PHOTOG_SW_HINTS)
    has_lidar_hint = any(h in sw_text for h in LIDAR_SW_HINTS)
    if has_nonzero:
        return "Likely LiDAR"
    if has_int_dim:
        return "Intensity field present"
    if has_lidar_hint and not has_photo_hint:
        return "Likely LiDAR by header hint"
    if has_photo_hint and not has_lidar_hint:
        return "Likely photogrammetry by header hint"
    return "Unknown"

def probe_file(path, env):
    ext = os.path.splitext(path)[1].lower()
    info = {
        "path": path,
        "file_ext": ext,
        "parent_dir": os.path.dirname(path),
        "project_key": project_key_for(path),
        "size": 0,
        "compressed": True if ext == ".laz" else False,
        "las_version": None,
        "point_format_id": None,
        "num_vlrs": None,
        "num_evlrs": None,
        "header_size": None,
        "offset_to_points": None,
        "point_count": None,
        "point_count_by_return": None,
        "scales": None,
        "offsets": None,
        "x_min": None, "x_max": None, "y_min": None, "y_max": None, "z_min": None, "z_max": None,
        "area_native": None,
        "unit_name": None,
        "unit_source": None,
        "area_m2": None,
        "density_pts_m2": None,
        "density_pts_native": None,
        "spacing_m": None,
        "has_rgb": None, "has_nir": None, "has_gps_time": None, "has_classification": None, "has_scan_angle": None,
        "extra_dim_count": None, "extra_dim_names": None,
        "system_identifier": None, "generating_software": None,
        "crs_name": None, "epsg": None,
        "intensity_present": None, "intensity_nonzero": None, "int_min": None, "int_max": None,
        "file_level_type": None,
        "error": None,
    }

    if ext == ".e57":
        try:
            st = os.stat(path)
            info["size"] = st.st_size
        except Exception as e:
            info["error"] = f"os.stat failed, {e}"
        info["file_level_type"] = "Unknown"
        info["error"] = info["error"] or "E57 parsing not implemented"
        return info

    try:
        st = os.stat(path)
        info["size"] = st.st_size
    except Exception as e:
        info["error"] = f"os.stat failed, {e}"
        return info

    if not env["laspy_ok"]:
        info["error"] = "laspy not installed"
        return info
    if ext == ".laz" and not env["lazrs_ok"]:
        info["error"] = "LAZ support missing, install lazrs"
        return info

    try:
        import laspy
        lp = win_long(path)
        with laspy.open(lp) as reader:
            hdr = reader.header

            # header core
            try:
                info["las_version"] = f"{hdr.version.major}.{hdr.version.minor}"
            except Exception:
                pass
            try:
                pf = getattr(hdr, "point_format", None)
                info["point_format_id"] = getattr(pf, "id", None)
            except Exception:
                pass
            info["num_vlrs"] = getattr(hdr, "number_of_vlrs", None)
            info["num_evlrs"] = getattr(hdr, "number_of_evlrs", None)
            info["header_size"] = getattr(hdr, "header_size", None)
            info["offset_to_points"] = getattr(hdr, "offset_to_point_data", None)

            # identity and CRS
            info["system_identifier"] = getattr(hdr, "system_identifier", None)
            info["generating_software"] = getattr(hdr, "generating_software", None)
            try:
                crs = hdr.parse_crs()
                if crs:
                    info["crs_name"] = getattr(crs, "name", None)
                    try:
                        info["epsg"] = crs.to_epsg()
                    except Exception:
                        info["epsg"] = None
            except Exception:
                pass

            # counts
            info["point_count"] = int(getattr(hdr, "point_count", 0))
            try:
                pbr = getattr(hdr, "point_count_by_return", None)
                if pbr is not None:
                    info["point_count_by_return"] = ",".join(str(int(v)) for v in pbr if int(v) >= 0)
            except Exception:
                pass

            # bounds and area
            x_min, x_max, y_min, y_max, z_min, z_max, area_native = _xy_area_from_header(hdr)
            info["x_min"], info["x_max"] = x_min, x_max
            info["y_min"], info["y_max"] = y_min, y_max
            info["z_min"], info["z_max"] = z_min, z_max
            info["area_native"] = area_native

            # dims presence
            try:
                dim_names = {d.name.lower() for d in hdr.point_format.dimensions}
            except Exception:
                dim_names = set()
            info["has_rgb"] = {"red","green","blue"}.issubset(dim_names)
            info["has_nir"] = "nir" in dim_names
            info["has_gps_time"] = "gps_time" in dim_names
            info["has_classification"] = "classification" in dim_names
            info["has_scan_angle"] = "scan_angle" in dim_names or "scan_angle_rank" in dim_names
            extra = dim_names - KNOWN_DIMS
            info["extra_dim_count"] = len(extra) if extra else 0
            info["extra_dim_names"] = ",".join(sorted(list(extra))[:8]) if extra else ""

            # intensity quick pass
            seen_nonzero = False
            i_min = None
            i_max = None
            sampled = 0
            for pts in reader.chunk_iterator(CHUNK_SIZE):
                if "intensity" in pts.array.dtype.names and pts.intensity.size:
                    cur_min = int(pts.intensity.min())
                    cur_max = int(pts.intensity.max())
                    i_min = cur_min if i_min is None else min(i_min, cur_min)
                    i_max = cur_max if i_max is None else max(i_max, cur_max)
                    if cur_max > 0:
                        seen_nonzero = True
                        break
                sampled += len(pts)
                if sampled >= SAMPLE_LIMIT_PER_FILE:
                    break
            info["intensity_present"] = "intensity" in dim_names
            info["intensity_nonzero"] = seen_nonzero
            info["int_min"] = i_min
            info["int_max"] = i_max

            # unit inference and densities
            path_lower = path.lower()
            unit_name, factor, source = _linear_unit_factor_to_m(reader, path_lower, x_min, x_max, y_min, y_max)
            info["unit_name"] = unit_name
            info["unit_source"] = source

            if area_native and info["point_count"]:
                info["density_pts_native"] = info["point_count"] / area_native

            if unit_name == "degree" and None not in (x_min, x_max, y_min, y_max) and info["point_count"]:
                lat_c = (y_min + y_max) / 2.0
                area_m2 = _deg_area_to_m2(abs(x_max - x_min), abs(y_max - y_min), lat_c)
                if area_m2 > 0:
                    info["area_m2"] = area_m2
                    info["density_pts_m2"] = info["point_count"] / area_m2
                    info["spacing_m"] = 1.0 / math.sqrt(info["density_pts_m2"]) if info["density_pts_m2"] and info["density_pts_m2"] > 0 else None
            elif area_native and info["point_count"] and factor:
                area_m2 = area_native * (factor ** 2)
                info["area_m2"] = area_m2
                if area_m2 > 0:
                    dens = info["point_count"] / area_m2
                    info["density_pts_m2"] = dens
                    info["spacing_m"] = 1.0 / math.sqrt(dens) if dens > 0 else None

            # file level classification
            gens = { (info["generating_software"] or "").lower() } if info.get("generating_software") else set()
            syst = { (info["system_identifier"] or "").lower() } if info.get("system_identifier") else set()
            info["file_level_type"] = _classify_lidar_like(
                has_nonzero=bool(info["intensity_nonzero"]),
                has_int_dim=bool(info["intensity_present"]),
                gens=gens,
                syst=syst
            )

            try:
                info["compressed"] = bool(getattr(reader, "compressed", info["compressed"]))
            except Exception:
                pass

    except Exception as e:
        info["error"] = str(e)

    return info

# rectangle union area in meters for one EPSG only
def _union_area_m2(rects):
    if not rects:
        return 0.0
    events = []
    for x1, x2, y1, y2 in rects:
        if x2 <= x1 or y2 <= y1:
            continue
        events.append((x1, 1, y1, y2))
        events.append((x2, -1, y1, y2))
    events.sort(key=lambda e: e[0])

    def merged_y_length(active):
        if not active:
            return 0.0
        segs = sorted(active)
        total = 0.0
        cur_s, cur_e = segs[0]
        for s, e in segs[1:]:
            if s > cur_e:
                total += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        total += cur_e - cur_s
        return total

    area = 0.0
    active = []
    prev_x = events[0][0]
    for x, typ, y1, y2 in events:
        dx = x - prev_x
        if dx > 0 and active:
            area += dx * merged_y_length(active)
        if typ == 1:
            active.append((y1, y2))
        else:
            try:
                active.remove((y1, y2))
            except ValueError:
                pass
        prev_x = x
    return max(0.0, area)

def summarize_project(project_dir, file_infos):
    files_ok = [fi for fi in file_infos if not fi.get("error")]
    total_size = sum(fi["size"] for fi in files_ok if fi.get("size") is not None)
    total_points = sum(fi["point_count"] for fi in files_ok if isinstance(fi.get("point_count"), int))

    # by extension counts
    ext_counter = Counter(fi["file_ext"] for fi in files_ok if fi.get("file_ext"))
    count_las = ext_counter.get(".las", 0)
    count_laz = ext_counter.get(".laz", 0)
    count_e57 = ext_counter.get(".e57", 0)

    # per file densities for averages
    dens_list = [fi["density_pts_m2"] for fi in files_ok if fi.get("density_pts_m2")]
    area_list = [fi["area_m2"] for fi in files_ok if fi.get("area_m2")]

    area_m2_sum = sum(area_list) if area_list else 0.0
    density_sum_method = (total_points / area_m2_sum) if total_points and area_m2_sum > 0 else None

    mean_density_arith = sum(dens_list) / len(dens_list) if dens_list else None
    mean_density_area_weighted = density_sum_method

    # union area in meters when EPSG is consistent
    epsg_set = {fi["epsg"] for fi in files_ok if fi.get("epsg") is not None}
    unit_types = {("degree" if fi.get("unit_name") == "degree" else "linear") for fi in files_ok if fi.get("unit_name")}
    union_area_m2 = None
    density_union_method = None
    overlap_ratio = None
    if len(epsg_set) == 1 and len(unit_types) == 1 and area_list:
        rects_m = []
        for fi in files_ok:
            if not fi.get("area_m2"):
                continue
            x1, x2, y1, y2 = fi.get("x_min"), fi.get("x_max"), fi.get("y_min"), fi.get("y_max")
            if None in (x1, x2, y1, y2):
                continue
            if fi.get("unit_name") == "degree":
                lat_c = (y1 + y2) / 2.0
                m_per_deg_lat = 111132.92 - 559.82 * math.cos(2*math.radians(lat_c)) + 1.175 * math.cos(4*math.radians(lat_c)) - 0.0023 * math.cos(6*math.radians(lat_c))
                m_per_deg_lon = 111412.84 * math.cos(math.radians(lat_c)) - 93.5 * math.cos(3*math.radians(lat_c)) + 0.118 * math.cos(5*math.radians(lat_c))
                X1 = x1 * m_per_deg_lon
                X2 = x2 * m_per_deg_lon
                Y1 = y1 * m_per_deg_lat
                Y2 = y2 * m_per_deg_lat
            else:
                if fi.get("area_native") and fi.get("area_m2") and fi["area_native"] > 0:
                    f = math.sqrt(fi["area_m2"] / fi["area_native"])
                else:
                    f = 1.0
                X1, X2, Y1, Y2 = x1 * f, x2 * f, y1 * f, y2 * f
            if X2 > X1 and Y2 > Y1:
                rects_m.append((X1, X2, Y1, Y2))
        if rects_m:
            union_area_m2 = _union_area_m2(rects_m)
            if union_area_m2 and total_points:
                density_union_method = total_points / union_area_m2
            if union_area_m2 and area_m2_sum:
                overlap_ratio = area_m2_sum / union_area_m2

    # intensity and z
    int_dim_files = sum(1 for fi in files_ok if fi.get("intensity_present"))
    nonzero_files = sum(1 for fi in files_ok if fi.get("intensity_nonzero"))
    mins = [fi["int_min"] for fi in files_ok if fi.get("int_min") is not None]
    maxs = [fi["int_max"] for fi in files_ok if fi.get("int_max") is not None]
    zmins = [fi.get("z_min") for fi in files_ok if fi.get("z_min") is not None]
    zmaxs = [fi.get("z_max") for fi in files_ok if fi.get("z_max") is not None]
    rep_min = min(mins) if mins else None
    rep_max = max(maxs) if maxs else None
    z_min_all = min(zmins) if zmins else None
    z_max_all = max(zmaxs) if zmaxs else None

    # returns rollup
    returns_sum = Counter()
    for fi in files_ok:
        pbr = fi.get("point_count_by_return")
        if pbr:
            try:
                vals = [int(x) for x in str(pbr).split(",") if x.strip() != ""]
                for i, v in enumerate(vals, start=1):
                    returns_sum[i] += v
            except Exception:
                pass
    returns_str = ";".join(f"r{i}={returns_sum[i]}" for i in sorted(returns_sum)) if returns_sum else ""

    unit_known_pct = round(100.0 * sum(1 for fi in files_ok if fi.get("area_m2")) / len(files_ok), 1) if files_ok else 0.0
    unit_set = {fi["unit_name"] for fi in files_ok if fi.get("unit_name")}
    pf_set = {fi["point_format_id"] for fi in files_ok if fi.get("point_format_id") is not None}
    epsg_str = ",".join(str(e) for e in sorted({fi["epsg"] for fi in files_ok if fi.get("epsg") is not None})) if files_ok else ""

    gens_set = {fi["generating_software"] for fi in files_ok if fi.get("generating_software")}
    syst_set = {fi["system_identifier"] for fi in files_ok if fi.get("system_identifier")}

    classification = _classify_lidar_like(
        has_nonzero=nonzero_files > 0,
        has_int_dim=int_dim_files > 0,
        gens={ (g or "").lower() for g in gens_set },
        syst={ (s or "").lower() for s in syst_set }
    )

    example_file = max(files_ok, key=lambda x: x.get("size", 0))["path"] if files_ok else ""
    error_msgs = [fi["error"] for fi in file_infos if fi.get("error")]
    first_error = error_msgs[0] if error_msgs else ""

    return {
        "project_directory": project_dir,
        "files_in_project": len(file_infos),
        "count_las": count_las,
        "count_laz": count_laz,
        "count_e57": count_e57,

        "total_size_human": human_size(total_size),
        "total_points": total_points,

        "files_with_intensity_dim": int_dim_files,
        "files_with_nonzero_intensity": nonzero_files,
        "intensity_min": rep_min,
        "intensity_max": rep_max,
        "z_min": z_min_all,
        "z_max": z_max_all,
        "intensity_range": "" if rep_min is None or rep_max is None else f"{rep_min} to {rep_max}",

        "area_m2_sum": round(area_m2_sum, 2) if area_m2_sum else "",
        "density_pts_m2_sum": round(density_sum_method, 3) if density_sum_method else "",
        "mean_density_arith": round(mean_density_arith, 3) if mean_density_arith else "",
        "mean_density_area_weighted": round(mean_density_area_weighted, 3) if mean_density_area_weighted else "",
        "union_epsg": next(iter(epsg_set)) if len(epsg_set) == 1 else "",
        "union_area_m2": round(union_area_m2, 2) if union_area_m2 else "",
        "density_pts_m2_union": round(density_union_method, 3) if density_union_method else "",
        "overlap_ratio_sum_div_union": round(overlap_ratio, 3) if overlap_ratio else "",

        "pct_with_known_units": unit_known_pct,
        "epsg_set": epsg_str,
        "point_format_ids": ",".join(str(p) for p in sorted(pf_set)) if pf_set else "",
        "units_set": ",".join(sorted(unit_set)) if unit_set else "",

        "returns_rollup": returns_str,
        "generating_software_set": ";".join(sorted(g for g in gens_set if g)),
        "system_identifier_set": ";".join(sorted(s for s in syst_set if s)),

        "classification": classification,
        "example_file": example_file,
        "files_with_errors": len(error_msgs),
        "first_error": first_error,
    }

def main():
    env = env_check()
    print("laspy installed,", env["laspy_ok"], ", version,", env["laspy_version"])
    print("LAZ support via lazrs,", env["lazrs_ok"])

    discovery_rows = []
    groups = defaultdict(list)

    all_paths = list(scan_lidar_files(SCAN_ROOT))
    print("Scanning root,", SCAN_ROOT)
    print("Discovered files,", len(all_paths))

    for fpath in all_paths:
        info = probe_file(fpath, env)
        discovery_rows.append({
            "path": info["path"],
            "file_ext": info["file_ext"],
            "parent_dir": info["parent_dir"],
            "project_key": info["project_key"],

            "size_bytes": info["size"],
            "compressed": info["compressed"],
            "las_version": info["las_version"] or "",
            "point_format_id": info["point_format_id"] if info["point_format_id"] is not None else "",
            "num_vlrs": info["num_vlrs"] if info["num_vlrs"] is not None else "",
            "num_evlrs": info["num_evlrs"] if info["num_evlrs"] is not None else "",
            "header_size": info["header_size"] if info["header_size"] is not None else "",
            "offset_to_points": info["offset_to_points"] if info["offset_to_points"] is not None else "",

            "point_count": info["point_count"] if info["point_count"] is not None else "",
            "point_count_by_return": info["point_count_by_return"] or "",

            "x_min": info["x_min"] if info["x_min"] is not None else "",
            "x_max": info["x_max"] if info["x_max"] is not None else "",
            "y_min": info["y_min"] if info["y_min"] is not None else "",
            "y_max": info["y_max"] if info["y_max"] is not None else "",
            "z_min": info["z_min"] if info["z_min"] is not None else "",
            "z_max": info["z_max"] if info["z_max"] is not None else "",

            "unit_name": info["unit_name"] or "",
            "unit_source": info["unit_source"] or "",
            "area_native": info["area_native"] if info["area_native"] is not None else "",
            "area_m2": info["area_m2"] if info["area_m2"] is not None else "",
            "density_pts_native": round(info["density_pts_native"], 3) if info["density_pts_native"] else "",
            "density_pts_m2": round(info["density_pts_m2"], 3) if info["density_pts_m2"] else "",
            "spacing_m": round(info["spacing_m"], 3) if info["spacing_m"] else "",

            "has_rgb": info["has_rgb"],
            "has_nir": info["has_nir"],
            "has_gps_time": info["has_gps_time"],
            "has_classification": info["has_classification"],
            "has_scan_angle": info["has_scan_angle"],
            "extra_dim_count": info["extra_dim_count"],
            "extra_dim_names": info["extra_dim_names"],

            "system_identifier": info["system_identifier"] or "",
            "generating_software": info["generating_software"] or "",
            "crs_name": info["crs_name"] or "",
            "epsg": info["epsg"] if info["epsg"] is not None else "",

            "intensity_present": info["intensity_present"],
            "intensity_nonzero": info["intensity_nonzero"],
            "int_min": info["int_min"] if info["int_min"] is not None else "",
            "int_max": info["int_max"] if info["int_max"] is not None else "",
            "intensity_range": "" if info["int_min"] is None or info["int_max"] is None else f"{info['int_min']} to {info['int_max']}",

            "file_level_type": info["file_level_type"] or "",
            "error": info["error"] or "",
        })
        groups[info["project_key"]].append(info)

    # discovery csv columns
    disc_fields = [
        "path","file_ext","parent_dir","project_key",
        "size_bytes","compressed",
        "las_version","point_format_id","num_vlrs","num_evlrs","header_size","offset_to_points",
        "point_count","point_count_by_return",
        "x_min","x_max","y_min","y_max","z_min","z_max",
        "unit_name","unit_source","area_native","area_m2","density_pts_native","density_pts_m2","spacing_m",
        "has_rgb","has_nir","has_gps_time","has_classification","has_scan_angle","extra_dim_count","extra_dim_names",
        "system_identifier","generating_software","crs_name","epsg",
        "intensity_present","intensity_nonzero","int_min","int_max","intensity_range",
        "file_level_type","error"
    ]
    os.makedirs(os.path.dirname(DISCOVERY_CSV), exist_ok=True)
    with open(DISCOVERY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=disc_fields)
        w.writeheader()
        for r in discovery_rows:
            w.writerow(r)
    print("Discovery CSV written,", DISCOVERY_CSV)

    # project rollups
    rows = []
    err_counter = Counter()
    for project_dir, infos in sorted(groups.items()):
        for fi in infos:
            if fi.get("error"):
                err_counter[fi["error"]] += 1
        rows.append(summarize_project(project_dir, infos))

    fields = [
        "project_directory","files_in_project","count_las","count_laz","count_e57",
        "total_size_human","total_points",
        "files_with_intensity_dim","files_with_nonzero_intensity",
        "intensity_min","intensity_max","intensity_range","z_min","z_max",
        "area_m2_sum","density_pts_m2_sum",
        "mean_density_arith","mean_density_area_weighted",
        "union_epsg","union_area_m2","density_pts_m2_union","overlap_ratio_sum_div_union",
        "pct_with_known_units","epsg_set","point_format_ids","units_set",
        "returns_rollup","generating_software_set","system_identifier_set",
        "classification","example_file","files_with_errors","first_error",
    ]
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("Project report written,", REPORT_CSV)

    print("\nProject level summary")
    header = ["Project directory","Files","LAS","LAZ","E57","Pts","Size","Type","Int range","Pts per m2 sum","Pts per m2 union","Overlap"]
    print("{:<92} {:>5} {:>4} {:>4} {:>4} {:>12} {:>12}  {:<22} {:<18} {:>15} {:>17} {:>8}".format(*header))
    for r in rows:
        print("{:<92} {:>5} {:>4} {:>4} {:>4} {:>12} {:>12}  {:<22} {:<18} {:>15} {:>17} {:>8}".format(
            r["project_directory"][:92],
            r["files_in_project"],
            r["count_las"], r["count_laz"], r["count_e57"],
            r["total_points"] if r["total_points"] else 0,
            r["total_size_human"],
            r["classification"][:22],
            r["intensity_range"][:18] if r["intensity_range"] else "n,a",
            r["density_pts_m2_sum"] if r["density_pts_m2_sum"] != "" else "n,a",
            r["density_pts_m2_union"] if r["density_pts_m2_union"] != "" else "n,a",
            r["overlap_ratio_sum_div_union"] if r["overlap_ratio_sum_div_union"] != "" else "n,a",
        ))

    print("\nTop error summaries")
    for msg, cnt in err_counter.most_common(10):
        print(cnt, ",", msg)

if __name__ == "__main__":
    main()
