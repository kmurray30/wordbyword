// Keeps the app pinned to the part of the screen you can actually see.
// On iOS the on-screen keyboard doesn't shrink the page; it scrolls the
// page up under the keyboard instead, which pushed the header off the top.
// The visual viewport is the visible part: publish its height and offset
// as CSS variables so .app (position: fixed, see App.css) always fills
// exactly that, with only the chat scrolling inside it.
export function trackVisibleViewport(): void {
  const vv = window.visualViewport;
  const root = document.documentElement;
  const update = () => {
    const height = vv ? vv.height : window.innerHeight;
    const top = vv ? vv.offsetTop : 0;
    root.style.setProperty("--vv-height", `${height}px`);
    root.style.setProperty("--vv-top", `${top}px`);
  };
  update();
  vv?.addEventListener("resize", update);
  vv?.addEventListener("scroll", update);
  window.addEventListener("resize", update);
}
