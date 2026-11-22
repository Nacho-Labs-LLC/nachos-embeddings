/**
 * Preload (`node --import ./scripts/block-network.mjs ...`) that makes any
 * outgoing network use fail loudly and records it. It patches the socket layer
 * (net.Socket#connect and dns lookups), which fetch/undici, http and https all
 * go through, so it catches requests no matter which client makes them.
 * Attempts are appended to the file named by $BLOCK_NETWORK_LOG, if set.
 */
import { appendFileSync } from "node:fs";
import dns from "node:dns";
import net from "node:net";

function blocked(what, target) {
  const entry = `${what} ${typeof target === "string" ? target : JSON.stringify(target)}`;
  if (process.env.BLOCK_NETWORK_LOG) {
    appendFileSync(process.env.BLOCK_NETWORK_LOG, entry + "\n");
  }
  return new Error(`network blocked: ${entry}`);
}

const connect = net.Socket.prototype.connect;
net.Socket.prototype.connect = function (...args) {
  const [first, second] = args;
  const isIpc =
    (typeof first === "string" && second === undefined) ||
    (typeof first === "object" && first !== null && "path" in first);
  if (isIpc) return connect.apply(this, args);
  throw blocked("connect", first);
};
for (const name of ["lookup", "resolve", "resolve4", "resolve6"]) {
  dns[name] = (host, ...rest) => {
    const callback = rest.find((r) => typeof r === "function");
    const error = blocked(`dns.${name}`, host);
    if (callback) process.nextTick(callback, error);
    else throw error;
  };
}
