import 'whatwg-fetch'

interface IStats {
  readonly complete: number;
  readonly total: number;
}

type PromptResult = false | ReadonlyArray<string>;

class App {
  constructor() {
  }

  public async run() {
    for await (const image of this.images()) {
      const stats = await this.stats();
      console.log(stats);

      const result = await this.prompt(image);

      if (result === false) {
        await this.skip(image);
      } else {
        await this.submit(image, result);
      }
    }
  }

  private async prompt(hash: string): Promise<PromptResult> {
    return false;
  }

  private async stats(): Promise<IStats> {
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

const app = new App();

app.run().catch((e) => {
  console.error(e);
});
