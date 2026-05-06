import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import BottomBar from "@/components/BottomBar";
import { ConnectionProvider } from "@/components/ConnectionProvider";

export default function ShellLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConnectionProvider>
      <Sidebar />
      <div className="ml-[148px] flex flex-col min-h-screen">
        <TopBar />
        <main className="mt-[56px] mb-8 h-[calc(100vh-56px-32px)] overflow-hidden">
          {children}
        </main>
        <BottomBar />
      </div>
    </ConnectionProvider>
  );
}
