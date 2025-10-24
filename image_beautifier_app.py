import os
import sys
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps, ImageChops


class BeautifyParams:
    """Container for beautification parameters with sane ranges."""

    def __init__(
        self,
        exposure: int = 0,  # -50..50 (percent-like)
        contrast: int = 0,  # -50..50
        saturation: int = 0,  # -50..50
        temperature: int = 0,  # -50..50 (negative=cooler, positive=warmer)
        clarity: int = 20,  # 0..100
        smooth: int = 15,  # 0..100
        denoise: int = 0,  # 0..100
        sharpen: int = 10,  # -20..80
    ) -> None:
        self.exposure = exposure
        self.contrast = contrast
        self.saturation = saturation
        self.temperature = temperature
        self.clarity = clarity
        self.smooth = smooth
        self.denoise = denoise
        self.sharpen = sharpen

    @classmethod
    def from_ui(cls, ui: "ImageBeautifierApp") -> "BeautifyParams":
        return cls(
            exposure=ui.var_exposure.get(),
            contrast=ui.var_contrast.get(),
            saturation=ui.var_saturation.get(),
            temperature=ui.var_temperature.get(),
            clarity=ui.var_clarity.get(),
            smooth=ui.var_smooth.get(),
            denoise=ui.var_denoise.get(),
            sharpen=ui.var_sharpen.get(),
        )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_preview_size(img_size: Tuple[int, int], max_edge: int = 1000) -> Tuple[int, int]:
    w, h = img_size
    if w <= max_edge and h <= max_edge:
        return w, h
    scale = max(w, h) / float(max_edge)
    return int(round(w / scale)), int(round(h / scale))


def apply_temperature(image: Image.Image, temperature: int) -> Image.Image:
    """Adjust color temperature by scaling R and B channels.

    temperature: -50..50 (negative=cooler, positive=warmer)
    """
    if temperature == 0:
        return image

    # Gains: small but noticeable. +/-50 => ~ +/-0.75x on R/B each side
    r_gain = clamp(1.0 + 0.015 * temperature, 0.25, 1.75)
    b_gain = clamp(1.0 - 0.015 * temperature, 0.25, 1.75)

    if image.mode != "RGB":
        work = image.convert("RGB")
    else:
        work = image

    r, g, b = work.split()

    def scale(c: Image.Image, gain: float) -> Image.Image:
        lut = [
            int(clamp(i * gain, 0, 255))
            for i in range(256)
        ]
        return c.point(lut)

    r2 = scale(r, r_gain)
    b2 = scale(b, b_gain)
    return Image.merge("RGB", (r2, g, b2))


def apply_s_curve(image: Image.Image, strength: float) -> Image.Image:
    """Apply a subtle S-curve to enhance midtones.

    strength: 0..1
    """
    if strength <= 0:
        return image

    def curve(x: float) -> float:
        # Smoothstep-inspired S-curve around 0.5, monotonic
        # Adjust curvature by strength
        t = x - 0.5
        y = 0.5 + t * (1 + 2 * strength) / (1 + abs(2 * t) * (2 - 2 * strength))
        return clamp(y, 0.0, 1.0)

    lut = [int(round(curve(i / 255.0) * 255.0)) for i in range(256)]
    if image.mode == "RGB":
        r, g, b = image.split()
        r = r.point(lut)
        g = g.point(lut)
        b = b.point(lut)
        return Image.merge("RGB", (r, g, b))
    return image.point(lut)


def edge_preserving_smooth(image: Image.Image, strength: int) -> Image.Image:
    """Approximate edge-preserving smoothing using blur + inverted edges as mask.

    strength: 0..100
    """
    if strength <= 0:
        return image

    radius = 0.0 + (strength / 100.0) * 3.5  # up to ~3.5px
    blurred = image.filter(ImageFilter.GaussianBlur(radius=radius))

    # Build mask from edges: higher mask => prefer blurred result on flat regions
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.invert(edges)
    # Normalize and scale mask strength
    mask = ImageEnhance.Contrast(edges).enhance(1.25)
    mask = ImageEnhance.Brightness(mask).enhance(1.1)

    # Scale mask by requested strength
    if strength < 100:
        # Multiply mask by alpha factor
        alpha = strength / 100.0
        lut = [int(i * alpha) for i in range(256)]
        mask = mask.point(lut)

    return Image.composite(blurred, image, mask)


def denoise_median(image: Image.Image, denoise: int) -> Image.Image:
    if denoise <= 0:
        return image
    if denoise < 40:
        return image.filter(ImageFilter.MedianFilter(size=3))
    elif denoise < 80:
        return image.filter(ImageFilter.MedianFilter(size=5))
    else:
        return image.filter(ImageFilter.MedianFilter(size=7))


def apply_clarity(image: Image.Image, clarity: int) -> Image.Image:
    """Local contrast enhancement via UnsharpMask with small radius.

    clarity: 0..100
    """
    if clarity <= 0:
        return image
    radius = 1.5
    percent = int(clamp(clarity * 2.0, 0, 400))  # up to 400%
    threshold = 2
    return image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))


def apply_sharpen(image: Image.Image, sharpen: int) -> Image.Image:
    if sharpen == 0:
        return image
    # Map -20..80 to factor ~ 0.6..1.8
    factor = clamp(1.0 + (sharpen / 100.0), 0.6, 1.8)
    return ImageEnhance.Sharpness(image).enhance(factor)


def apply_linear_enhance(value: int, base: float = 1.0, scale: float = 0.02) -> float:
    """Map integer slider values to multiplicative factor.

    For -50..50 with scale=0.02 => 0.0..2.0 around 1.0 center.
    """
    return clamp(base + value * scale, 0.0, 3.0)


def beautify_pipeline(image: Image.Image, params: BeautifyParams) -> Image.Image:
    """Apply a series of enhancements to produce a pleasing result."""
    work = image

    # 1) Denoise (before most enhancements)
    work = denoise_median(work, params.denoise)

    # 2) Edge-preserving smooth (skin smoothing feel)
    work = edge_preserving_smooth(work, params.smooth)

    # 3) Exposure/Brightness
    if params.exposure != 0:
        work = ImageEnhance.Brightness(work).enhance(apply_linear_enhance(params.exposure, scale=0.02))

    # 4) Contrast
    if params.contrast != 0:
        work = ImageEnhance.Contrast(work).enhance(apply_linear_enhance(params.contrast, scale=0.02))

    # 5) Saturation (Color)
    if params.saturation != 0:
        work = ImageEnhance.Color(work).enhance(apply_linear_enhance(params.saturation, scale=0.02))

    # 6) Color temperature
    work = apply_temperature(work, params.temperature)

    # 7) Gentle S-curve tied subtly to contrast increase
    curve_strength = clamp((params.contrast + 20) / 120.0, 0.0, 0.35)
    work = apply_s_curve(work, curve_strength)

    # 8) Clarity (local contrast)
    work = apply_clarity(work, params.clarity)

    # 9) Global sharpen at the end
    work = apply_sharpen(work, params.sharpen)

    return work


class ImageBeautifierApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Image Beautifier")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 650)

        # State
        self.original_image: Optional[Image.Image] = None
        self.preview_base_image: Optional[Image.Image] = None
        self.preview_image_tk: Optional[ImageTk.PhotoImage] = None
        self.preview_max_edge: int = 1000
        self.current_preview: Optional[Image.Image] = None
        self.update_job: Optional[str] = None
        self.image_path: Optional[str] = None

        # Variables (sliders)
        self.var_exposure = tk.IntVar(value=0)
        self.var_contrast = tk.IntVar(value=0)
        self.var_saturation = tk.IntVar(value=0)
        self.var_temperature = tk.IntVar(value=0)
        self.var_clarity = tk.IntVar(value=20)
        self.var_smooth = tk.IntVar(value=15)
        self.var_denoise = tk.IntVar(value=0)
        self.var_sharpen = tk.IntVar(value=10)

        self._build_ui()

    # UI building
    def _build_ui(self) -> None:
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=(10, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        btn_open = ttk.Button(toolbar, text="Open Image", command=self.on_open)
        btn_open.pack(side=tk.LEFT, padx=(0, 6))

        btn_save = ttk.Button(toolbar, text="Save As...", command=self.on_save_as)
        btn_save.pack(side=tk.LEFT, padx=(0, 6))

        btn_reset = ttk.Button(toolbar, text="Reset", command=self.on_reset)
        btn_reset.pack(side=tk.LEFT, padx=(0, 6))

        preset_menu = ttk.Menubutton(toolbar, text="Presets")
        menu = tk.Menu(preset_menu, tearoff=0)
        menu.add_command(label="Natural Boost", command=self.preset_natural)
        menu.add_command(label="Portrait Smooth", command=self.preset_portrait)
        menu.add_command(label="Vibrant", command=self.preset_vibrant)
        menu.add_command(label="Crisp & Cool", command=self.preset_crisp_cool)
        preset_menu["menu"] = menu
        preset_menu.pack(side=tk.LEFT, padx=(0, 6))

        # Main layout: left preview, right controls
        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Preview area with canvas
        preview_frame = ttk.Frame(main, padding=10)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(preview_frame, background="#222", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Controls panel
        controls = ttk.Frame(main, padding=10)
        controls.pack(side=tk.RIGHT, fill=tk.Y)

        def add_slider(parent, label: str, var: tk.IntVar, from_, to, command=None):
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=4)
            ttk.Label(frame, text=label).pack(anchor=tk.W)
            scale = ttk.Scale(
                frame,
                from_=from_,
                to=to,
                orient=tk.HORIZONTAL,
                command=(lambda _evt: self.on_params_changed()) if command is None else command,
                variable=var
            )
            # ttk.Scale has no direct integer tick display; add linked entry showing value
            scale.pack(fill=tk.X)
            value_row = ttk.Frame(frame)
            value_row.pack(fill=tk.X)
            val_label = ttk.Label(value_row, textvariable=var, width=5)
            val_label.pack(side=tk.RIGHT)
            return scale

        add_slider(controls, "Exposure", self.var_exposure, -50, 50)
        add_slider(controls, "Contrast", self.var_contrast, -50, 50)
        add_slider(controls, "Saturation", self.var_saturation, -50, 50)
        add_slider(controls, "Temperature", self.var_temperature, -50, 50)
        add_slider(controls, "Clarity", self.var_clarity, 0, 100)
        add_slider(controls, "Smooth Skin", self.var_smooth, 0, 100)
        add_slider(controls, "Denoise", self.var_denoise, 0, 100)
        add_slider(controls, "Sharpen", self.var_sharpen, -20, 80)

        # Footer hint
        footer = ttk.Frame(self.root, padding=(10, 6))
        footer.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Label(footer, text="Tip: try a preset, then fine-tune with sliders.").pack(anchor=tk.W)

    # Event handlers
    def on_open(self) -> None:
        filetypes = [
            ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Open image", filetypes=filetypes)
        if not path:
            return

        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not open image: {exc}")
            return

        self.image_path = path
        self.original_image = img
        self._prepare_preview_base()
        self.on_reset()  # reset sliders; will trigger a preview update

    def on_save_as(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        filetypes = [
            ("JPEG", "*.jpg;*.jpeg"),
            ("PNG", "*.png"),
            ("WebP", "*.webp"),
            ("TIFF", "*.tif;*.tiff"),
            ("BMP", "*.bmp"),
        ]
        default_name = "enhanced.jpg"
        initialdir = os.path.dirname(self.image_path) if self.image_path else os.getcwd()
        out_path = filedialog.asksaveasfilename(
            title="Save enhanced image as",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=filetypes,
            initialdir=initialdir,
        )
        if not out_path:
            return

        params = BeautifyParams.from_ui(self)
        try:
            full_result = beautify_pipeline(self.original_image, params)
            # Choose save format based on extension
            ext = os.path.splitext(out_path)[1].lower()
            save_kwargs = {}
            if ext in (".jpg", ".jpeg"):
                save_kwargs = {"quality": 92, "optimize": True}
            elif ext == ".png":
                save_kwargs = {"compress_level": 6}
            elif ext == ".webp":
                save_kwargs = {"quality": 90}

            full_result.save(out_path, **save_kwargs)
            messagebox.showinfo("Saved", f"Saved enhanced image to:\n{out_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", f"Could not save image: {exc}")

    def on_reset(self) -> None:
        # Defaults tuned for subtle improvements
        self.var_exposure.set(0)
        self.var_contrast.set(8)
        self.var_saturation.set(8)
        self.var_temperature.set(0)
        self.var_clarity.set(20)
        self.var_smooth.set(15)
        self.var_denoise.set(0)
        self.var_sharpen.set(12)
        self.on_params_changed()

    def on_params_changed(self) -> None:
        # Debounce preview updates for smoother UI while dragging
        if self.update_job is not None:
            self.root.after_cancel(self.update_job)
            self.update_job = None
        self.update_job = self.root.after(120, self._update_preview)

    def _prepare_preview_base(self) -> None:
        if self.original_image is None:
            self.preview_base_image = None
            return
        max_edge = self._compute_canvas_max_edge()
        size = compute_preview_size(self.original_image.size, max_edge=max_edge)
        self.preview_base_image = self.original_image.resize(size, resample=Image.LANCZOS)

    def _compute_canvas_max_edge(self) -> int:
        # Estimate how big we can render inside the canvas with some padding
        c_width = max(200, int(self.canvas.winfo_width()))
        c_height = max(200, int(self.canvas.winfo_height()))
        max_edge = int(max(min(c_width, c_height) - 40, 400))
        return max(400, min(1500, max_edge))

    def _on_canvas_resize(self, _event) -> None:
        # When canvas changes size, recompute preview base to fit nicely
        if self.original_image is None:
            return
        self._prepare_preview_base()
        self.on_params_changed()

    def _update_preview(self) -> None:
        self.update_job = None
        if self.preview_base_image is None:
            # Clear canvas
            self.canvas.delete("all")
            return

        params = BeautifyParams.from_ui(self)
        try:
            result = beautify_pipeline(self.preview_base_image, params)
        except Exception as exc:
            messagebox.showerror("Preview error", f"Failed to render preview: {exc}")
            return

        self.current_preview = result
        self._render_on_canvas(result)

    def _render_on_canvas(self, img: Image.Image) -> None:
        # Letterbox the image centered on canvas
        c_width = int(self.canvas.winfo_width())
        c_height = int(self.canvas.winfo_height())
        if c_width <= 2 or c_height <= 2:
            return

        # Fit to canvas (already near the right size, but ensure it fits)
        scale = min((c_width - 20) / img.width, (c_height - 20) / img.height)
        scale = clamp(scale, 0.1, 3.0)
        target_size = (int(img.width * scale), int(img.height * scale))
        shown = img if target_size == img.size else img.resize(target_size, Image.LANCZOS)

        self.preview_image_tk = ImageTk.PhotoImage(shown)
        self.canvas.delete("all")
        x = (c_width - shown.width) // 2
        y = (c_height - shown.height) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.preview_image_tk)

    # Presets
    def preset_natural(self) -> None:
        self.var_exposure.set(4)
        self.var_contrast.set(10)
        self.var_saturation.set(10)
        self.var_temperature.set(4)
        self.var_clarity.set(22)
        self.var_smooth.set(12)
        self.var_denoise.set(0)
        self.var_sharpen.set(10)
        self.on_params_changed()

    def preset_portrait(self) -> None:
        self.var_exposure.set(6)
        self.var_contrast.set(6)
        self.var_saturation.set(8)
        self.var_temperature.set(8)
        self.var_clarity.set(12)
        self.var_smooth.set(28)
        self.var_denoise.set(10)
        self.var_sharpen.set(6)
        self.on_params_changed()

    def preset_vibrant(self) -> None:
        self.var_exposure.set(0)
        self.var_contrast.set(16)
        self.var_saturation.set(22)
        self.var_temperature.set(2)
        self.var_clarity.set(30)
        self.var_smooth.set(10)
        self.var_denoise.set(0)
        self.var_sharpen.set(18)
        self.on_params_changed()

    def preset_crisp_cool(self) -> None:
        self.var_exposure.set(0)
        self.var_contrast.set(14)
        self.var_saturation.set(6)
        self.var_temperature.set(-8)
        self.var_clarity.set(28)
        self.var_smooth.set(10)
        self.var_denoise.set(0)
        self.var_sharpen.set(20)
        self.on_params_changed()


def main() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        # Headless environments may fail to create a Tk window
        sys.stderr.write(
            "Failed to initialize Tk GUI (are you running headless?).\n"
        )
        sys.stderr.write(str(exc) + "\n")
        sys.exit(1)

    # Use platform-appropriate theming
    try:
        style = ttk.Style()
        if sys.platform == "darwin":
            style.theme_use("aqua")
        else:
            # Use a modern theme if available
            for candidate in ("clam", "default"):
                try:
                    style.theme_use(candidate)
                    break
                except tk.TclError:
                    continue
    except Exception:
        pass

    app = ImageBeautifierApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
