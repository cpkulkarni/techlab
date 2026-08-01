import asyncio
import os
import sys

class SimpleSMTPServer:
    def __init__(self, host='127.0.0.1', port=1025):
        self.host = host
        self.port = port

    async def handle_client(self, reader, writer):
        writer.write(b"220 Simple Python SMTP Server Ready\r\n")
        await writer.drain()

        mail_from = ""
        rcpt_tos = []
        data_mode = False
        data_lines = []

        try:
            while True:
                line_bytes = await reader.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode('utf-8', errors='ignore').strip()

                if data_mode:
                    if line == '.':
                        data_mode = False
                        email_content = "\n".join(data_lines)
                        print(f"\n--- RECEIVED EMAIL ---")
                        print(f"From: {mail_from}")
                        print(f"To: {', '.join(rcpt_tos)}")
                        print(f"Content:\n{email_content}")
                        print(f"----------------------\n")
                        sys.stdout.flush()
                        
                        # Save to log file
                        os.makedirs("app-log", exist_ok=True)
                        with open(os.path.join("app-log", "last_python_email.txt"), "w", encoding="utf-8") as f:
                            f.write(f"From: {mail_from}\nTo: {', '.join(rcpt_tos)}\n\n{email_content}")

                        writer.write(b"250 OK Message accepted\r\n")
                        await writer.drain()
                        data_lines = []
                    else:
                        data_lines.append(line)
                    continue

                upper_line = line.upper()
                if upper_line.startswith("HELO") or upper_line.startswith("EHLO"):
                    writer.write(b"250 Hello\r\n")
                elif upper_line.startswith("MAIL FROM:"):
                    mail_from = line[10:].strip("<> ")
                    writer.write(b"250 2.1.0 Ok\r\n")
                elif upper_line.startswith("RCPT TO:"):
                    rcpt_tos.append(line[8:].strip("<> "))
                    writer.write(b"250 2.1.5 Ok\r\n")
                elif upper_line == "DATA":
                    data_mode = True
                    writer.write(b"354 End data with <CR><LF>.<CR><LF>\r\n")
                elif upper_line == "QUIT":
                    writer.write(b"221 Bye\r\n")
                    await writer.drain()
                    break
                elif upper_line == "NOOP":
                    writer.write(b"250 Ok\r\n")
                elif upper_line == "RSET":
                    mail_from = ""
                    rcpt_tos = []
                    data_mode = False
                    data_lines = []
                    writer.write(b"250 Ok\r\n")
                else:
                    writer.write(b"500 Command not recognized\r\n")
                await writer.drain()
        except Exception as e:
            print(f"Error handling client: {e}")
            sys.stdout.flush()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self):
        server = await asyncio.start_server(self.handle_client, self.host, self.port)
        print(f"Python SMTP server running on {self.host}:{self.port}")
        sys.stdout.flush()
        async with server:
            await server.serve_forever()

if __name__ == '__main__':
    port = 1025
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    asyncio.run(SimpleSMTPServer(port=port).start())
