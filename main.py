import sys
import numpy as np
import pygame
import pygame.surfarray


PALETTES = [
    ((0.50, 0.50, 0.50), (0.50, 0.50, 0.50), (1.00, 1.00, 1.00), (0.00, 0.10, 0.20)),
    ((0.50, 0.50, 0.50), (0.50, 0.50, 0.50), (1.00, 1.00, 1.00), (0.00, 0.33, 0.67)),
    ((0.50, 0.50, 0.50), (0.50, 0.50, 0.50), (2.00, 1.00, 0.00), (0.50, 0.20, 0.25)),
    ((0.80, 0.50, 0.40), (0.20, 0.40, 0.20), (1.00, 1.00, 1.00), (0.00, 0.25, 0.25)),
]

DEFAULT_CENTER = complex(-0.6, 0.0)
DEFAULT_SCALE = 3.2
MIN_SCALE = 1e-14
MAX_SCALE = 10.0


def mandelbrot_smooth(width, height, xmin, xmax, ymin, ymax, max_iter):
    x = np.linspace(xmin, xmax, width, dtype=np.float64)
    y = np.linspace(ymax, ymin, height, dtype=np.float64)

    C = (x[np.newaxis, :] + 1j * y[:, np.newaxis]).ravel()
    n = C.size
    Z = np.zeros(n, dtype=np.complex128)
    it = np.full(n, float(max_iter + 1), dtype=np.float64)

    alive = np.arange(n)
    for i in range(max_iter):
        Za = Z[alive] * Z[alive] + C[alive]
        Z[alive] = Za

        mag2 = Za.real * Za.real + Za.imag * Za.imag
        escaped = mag2 > 4.0

        if escaped.any():
            it[alive[escaped]] = i + 1
            alive = alive[~escaped]
            if alive.size == 0:
                break
    interior = it > max_iter
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = it + 1.0 - np.log(np.log(np.abs(Z))) / np.log(2.0)

    mu = np.nan_to_num(mu, nan=float(max_iter), posinf=float(max_iter), neginf=0.0)
    mu[interior] = max_iter
    return mu.reshape(height, width), interior.reshape(height, width)


def colorize(mu, interior, max_iter, palette_idx):
    a, b, c, d = (np.asarray(v, dtype=np.float64) for v in PALETTES[palette_idx])
    t = np.clip(mu / max_iter, 0.0, 1.0) ** 0.5
    rgb = a + b * np.cos(2.0 * np.pi * (c * t[..., np.newaxis] + d))
    rgb = np.clip(rgb, 0.0, 1.0)
    rgb[interior] = 0.0
    return (rgb * 255.0 + 0.5).astype(np.uint8)


class MandelbrotApp:

    def __init__(self, width=1000, height=750):
        pygame.init()
        pygame.display.set_caption("Mandelbrot Set Explorer")
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)

        self.center = DEFAULT_CENTER
        self.scale = DEFAULT_SCALE
        self.max_iter = 200
        self.palette_idx = 0
        self.dragging = False
        self.needs_render = True
        self.running = True
        self.show_intro()

    def show_intro(self):
        w, h = self.screen.get_size()
        self.screen.fill((12, 12, 18))

        font = pygame.font.SysFont("consolas,couriernew,monospace", 20)
        lines = [
            "scroll for zoom and right click drag for pan",
            "use P to change colour palette",
            "",
            "click or press any key to begin",
        ]
        rendered = [font.render(line, True, (220, 220, 220)) for line in lines]
        total = sum(r.get_height() + 8 for r in rendered)
        y = (h - total) // 2
        for r in rendered:
            self.screen.blit(r, ((w - r.get_width()) // 2, y))
            y += r.get_height() + 8
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                elif event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    waiting = False
    def view_bounds(self, w, h):
        half_w = self.scale * 0.5
        half_h = half_w * (h / w)
        return (self.center.real - half_w, self.center.real + half_w,
                self.center.imag - half_h, self.center.imag + half_h)

    def render(self, w, h):
        xmin, xmax, ymin, ymax = self.view_bounds(w, h)
        mu, interior = mandelbrot_smooth(w, h, xmin, xmax, ymin, ymax, self.max_iter)
        rgb = colorize(mu, interior, self.max_iter, self.palette_idx)
        return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.transpose(1, 0, 2)))

    def redraw(self):
        w, h = self.screen.get_size()
        if self.dragging:
            pw, ph = max(w // 4, 1), max(h // 4, 1)
            preview = self.render(pw, ph)
            self.screen.blit(pygame.transform.smoothscale(preview, (w, h)), (0, 0))
        else:
            self.screen.blit(self.render(w, h), (0, 0))
        pygame.display.flip()

    def zoom_at(self, pos, factor):
        w, h = self.screen.get_size()
        mx, my = pos

        new_scale = self.scale / factor
        if not (MIN_SCALE <= new_scale <= MAX_SCALE):
            return

        aspect = h / w
        anchor_x = self.center.real + (mx / w - 0.5) * self.scale
        anchor_y = self.center.imag - (my / h - 0.5) * self.scale * aspect

        self.center = complex(
            anchor_x - (mx / w - 0.5) * new_scale,
            anchor_y + (my / h - 0.5) * new_scale * aspect,
        )
        self.scale = new_scale
        zoom = DEFAULT_SCALE / self.scale
        self.max_iter = int(np.clip(200 + 150 * np.log10(max(1.0, zoom)), 200, 3000))
        self.needs_render = True
    def pan(self, rel):
        dx, dy = rel
        unit = self.scale / self.screen.get_width()
        self.center = complex(self.center.real - dx * unit,
                              self.center.imag + dy * unit)
        self.needs_render = True
    def handle_events(self):
        has_wheel = hasattr(pygame, "MOUSEWHEEL")
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_p:
                    self.palette_idx = (self.palette_idx + 1) % len(PALETTES)
                    self.needs_render = True
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 3:
                    self.dragging = True
                elif not has_wheel and event.button in (4, 5):
                    self.zoom_at(event.pos, 1.25 if event.button == 4 else 0.8)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3 and self.dragging:
                    self.dragging = False
                    self.needs_render = True
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    self.pan(event.rel)
            elif has_wheel and event.type == pygame.MOUSEWHEEL:
                self.zoom_at(pygame.mouse.get_pos(), 1.2 ** event.y)
            elif event.type == pygame.VIDEORESIZE:
                self.needs_render = True

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            self.handle_events()
            if self.needs_render:
                self.needs_render = False
                self.redraw()
            clock.tick(60)
        pygame.quit()
        sys.exit(0)


def main():
    MandelbrotApp(1000, 750).run()


if __name__ == "__main__":
    main()