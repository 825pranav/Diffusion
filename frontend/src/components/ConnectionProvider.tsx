"use client";

import { createContext, useContext, useState, useCallback } from "react";

interface ConnectionCtx {
  connected: boolean;
  setConnected: (v: boolean) => void;
}

const ConnectionContext = createContext<ConnectionCtx>({
  connected: false,
  setConnected: () => {},
});

export function ConnectionProvider({ children }: { children: React.ReactNode }) {
  const [connected, setConnected] = useState(false);
  const set = useCallback((v: boolean) => setConnected(v), []);
  return (
    <ConnectionContext.Provider value={{ connected, setConnected: set }}>
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  return useContext(ConnectionContext);
}
