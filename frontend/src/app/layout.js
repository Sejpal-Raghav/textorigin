import './globals.css';

export const metadata = {
  title: 'TextOrigin',
  description: 'Advanced AI vs Human Text Classifier',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
