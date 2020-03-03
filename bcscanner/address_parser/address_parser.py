# Copyright Philip Morris Products S.A. 2019


try:
    from postal.parser import parse_address as pa

    def parse_address(text):
        result = pa(text)
        return [{key: val} for val, key in result]

except ImportError:

    import json
    import os

    import requests
    import docker
    import atexit
    import time
    import socket

    def wait_for_port(port, host='localhost', timeout=300.0):

        wait_for_port.was_checked = getattr(wait_for_port, 'was_checked', False)
        if wait_for_port.was_checked:
            return

        start_time = time.perf_counter()
        while True:
            try:
                requests.get("http://localhost:8080")
                wait_for_port.was_checked = True
                break
            except requests.exceptions.ConnectionError as ex:
                time.sleep(0.01)
                if time.perf_counter() - start_time >= timeout:
                    raise TimeoutError('Waited too long for the port {} on host {} to start accepting '
                                       'connections.'.format(port, host)) from ex

    print("INFO: libpostal is not available on local system: running using docker")
    client = docker.from_env()
    image, _ = client.images.build(path=os.path.dirname(__file__))
    container = client.containers.run(image, "--server", ports={8080: 8080}, detach=True)

    def parse_address(text):

        wait_for_port(8080)

        response = requests.post("http://localhost:8080", json={"text": text})

        return response.json()

    def exit_handler():
        global container
        print("Info: stopping libpostal from docker")
        container.stop()

    atexit.register(exit_handler)


if __name__ == "__main__":

    import sys
    import json

    if len(sys.argv) == 1:

        test_data = [
            [
                "Member of",
                "APS",
                "Physics",
                "www.aps.org",
                "Remi Zaidan",
                "Postdoctoral Research Assistant",
                "Email: zaidan@cern.ch",
                "Phone: (0033) 669749919",
                "The University of Iowa",
                "Office: CERN, CH-1201",
                "Geneva 23, Switzerland",
                "American Physics Society"
            ]
        ]

        for lines in test_data:
            response = parse_address("\n".join(lines))

            print(response)

    else:

        if "--server" in sys.argv:

            print("INFO: Called with server option")

            from http.server import BaseHTTPRequestHandler, HTTPServer

            class S(BaseHTTPRequestHandler):

                def _set_headers(self, content_type='text/html'):
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.end_headers()

                def do_GET(self):
                    self._set_headers()
                    self.wfile.write("It's Alive".encode('utf-8'))

                def do_HEAD(self):
                    self._set_headers()

                def do_POST(self):
                    print("SERVER: POST Called")
                    content_length = int(self.headers['Content-Length'])
                    post_data = json.loads(self.rfile.read(content_length))
                    parsed = parse_address(post_data["text"])
                    self._set_headers(content_type='application/json')
                    self.wfile.write(json.dumps(parsed).encode('utf-8'))
                    print("SERVER: POST request processed")

            server_class = HTTPServer
            handler_class = S
            port = 8080
            server_address = ('0.0.0.0', port)
            httpd = server_class(server_address, handler_class)
            print('Info: Starting libpostal on port: {}'.format(port))
            httpd.serve_forever()
        else:
            print(json.dumps(parse_address(" ".join(sys.argv[1:]))))
