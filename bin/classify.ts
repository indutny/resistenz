#!/usr/bin/env npx ts-node

import { Server } from '../src/classify';

const server = new Server();

server.listen(8000, () => {
});
