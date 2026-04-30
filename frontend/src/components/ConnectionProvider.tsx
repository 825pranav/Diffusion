"use client";

import { createContext, useContext, useState } from "react";

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
  return (
    <ConnectionContext.Provider value={{ connected, setConnected }}>
      {children}
    </ConnectionContext.Provider>
  );
}

export function useConnection() {
  return useContext(ConnectionContext);
}
