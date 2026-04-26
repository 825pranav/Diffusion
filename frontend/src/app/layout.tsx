import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import { ConnectionProvider } from "@/components/ConnectionProvider";

const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});

export const metadata: Metadata = {
  title: "Diffusion",
  description: "Distributed trend propagation engine",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistMono.variable} antialiased bg-surface text-foreground`}>
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
      </body>
    </html>
  );
}
