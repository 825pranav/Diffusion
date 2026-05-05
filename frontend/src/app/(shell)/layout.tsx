import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import { ConnectionProvider } from "@/components/ConnectionProvider";

export default function ShellLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConnectionProvider>
      <div className="flex h-screen w-screen overflow-hidden">
        <Sidebar />
        <div className="flex flex-col flex-1 min-w-0">
          <TopBar />
          <main className="flex-1 overflow-hidden relative">
            {children}
          </main>
        </div>
      </div>
    </ConnectionProvider>
  );
}
