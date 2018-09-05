import 'whatwg-fetch'

interface IStats {
  readonly complete: number;
  readonly total: number;
}

type PromptResult = false | ReadonlyArray<string>;

const KEY_TO_COLOR_MAP = new Map([
  [ 'b', 'black' ],
  [ 'e', 'brown' ],
  [ 'r', 'red' ],
  [ 'o', 'orange' ],
  [ 'y', 'yellow' ],
  [ 'g', 'green' ],
  [ 'l', 'blue' ],
  [ 'v', 'violet' ],
  [ 'a', 'grey' ],
  [ 'w', 'white' ],
  [ 'z', 'gold' ],
  [ 'x', 'silver' ],
  [ ' ', 'none' ],
]);

class App {
  private image: HTMLImageElement | undefined;
  private colorBar: HTMLElement;
  private stats: HTMLElement;
  private keyHandler: ((key: string) => boolean) | undefined;

  constructor(private readonly container: HTMLElement) {
    window.addEventListener('keypress', (e) => {
      if (this.keyHandler && !this.keyHandler(e.key)) {
        e.preventDefault();
        return false;
      }
    }, true);

    // Stats
    const stats = document.createElement('section');
    stats.className = 'stats';

    this.container.appendChild(stats);
    this.stats = stats;

    // Colors
    const colorBar = document.createElement('section');
    colorBar.className = 'colorbar';

    this.container.appendChild(colorBar);
    this.colorBar = colorBar;

    // Legend
    this.container.appendChild(this.generateLegend());
  }

  public async run() {
    for await (const image of this.images()) {
      const stats = await this.fetchStats();

      this.stats.textContent = JSON.stringify(stats);

      const result = await this.prompt(image);

      if (result === false) {
        await this.skip(image);
      } else {
        await this.submit(image, result);
      }
    }
  }

  private generateLegend(): HTMLElement {
    const res = document.createElement('section');
    res.className = 'legend';

    const header = document.createElement('h3');
    header.textContent = 'Key codes';
    res.appendChild(header);

    const ul = document.createElement('ul');
    res.appendChild(ul);

    for (const [ key, value ] of KEY_TO_COLOR_MAP) {
      const li = document.createElement('li');

      const keyWrap = document.createElement('b');
      keyWrap.textContent = `"${key}"`;

      const valueWrap = document.createElement('span');
      valueWrap.textContent = ` - ${value}`;

      li.appendChild(keyWrap);
      li.appendChild(valueWrap);

      ul.appendChild(li);
    }

    return res;
  }

  private async prompt(hash: string): Promise<PromptResult> {
    const image = await this.loadImage(hash);

    if (this.image) {
      this.container.removeChild(this.image);
    }
    this.container.insertBefore(image, this.colorBar);
    this.image = image;

    this.colorBar.innerHTML = '';

    const colors: string[] = [];
    const elems: HTMLElement[] = [];

    const addColor = (color: string) => {
      const elem = document.createElement('span');
      elem.textContent = color;
      elem.className = color;

      this.colorBar.appendChild(elem);

      colors.push(color);
      elems.push(elem);
    };

    const popColor = () => {
      const elem = elems.pop();
      if (elem) {
        elem.remove();
      }
      colors.pop();
    };

    return new Promise<PromptResult>((resolve) => {
      this.keyHandler = (key: string) => {
        if (KEY_TO_COLOR_MAP.has(key)) {
          addColor(KEY_TO_COLOR_MAP.get(key)!);
        } else if (key === 'Backspace') {
          popColor();
        } else if (key === 'Enter') {
          // Submit
          this.keyHandler = undefined;
          resolve(colors);
        } else if (key === 'Escape') {
          // Skip
          resolve(false);
        } else {
          // Ignore
          return true;
        }
        return false;
      };
    });
  }

  private async fetchStats(): Promise<IStats> {
    const res = await fetch('/api/stats');
    return await res.json();
  }

  private async *images() {
    for (;;) {
      const res = await fetch('/api/next');
      const json = await res.json();
      if (json.done) {
        return;
      }

      yield json.hash;
    }
  }

  private async loadImage(hash: string): Promise<HTMLImageElement> {
    const img = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = document.createElement('img') as HTMLImageElement;

      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));

      img.src = `/images/${hash}.jpg`;
    });

    // Downscale
    const max = Math.max(img.width, img.height);
    const scale = 256 / max;

    img.width *= scale;
    img.height *= scale;

    return img;
  }

  private async skip(hash: string) {
    const res = await fetch(`/api/label/${hash}`, { method: 'DELETE' });
    return await res.json();
  }

  private async submit(hash: string, colors: ReadonlyArray<string>) {
    await fetch(`/api/label/${hash}`, {
      method: 'PUT',
      headers: {
        'content-type': 'application/json',
      },
      body: JSON.stringify({ colors }),
    });
  }
}

const app = new App(document.getElementById('content')!);

app.run().catch((e) => {
  console.error(e);
});
