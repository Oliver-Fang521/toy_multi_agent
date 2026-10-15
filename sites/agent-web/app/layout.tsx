import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Toy Multi Agent — Research with receipts",
  description: "Ask a question. Follow the research. Get an answer backed by sources.",
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">{children}</body>
    </html>
  );
}
