import { homedir } from "os";
import { join } from "path";

export const CONFIG = {
  sessionDir: process.env.AZOR_SESSION_DIR || join(homedir(), ".azor"),
  walFilename: "azor-wal.json",
} as const;
