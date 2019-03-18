import sys
import os
import time
import platform
import atexit
from signal import SIGTERM


class EasyDaemon:
    """
    A generic daemon class.

    Usage: subclass the Daemon class and override the run() method
    """

    def __init__(self, daemon_conf):
        self.daemon_conf = daemon_conf
        self.stdin = daemon_conf['stdin']
        self.stdout = daemon_conf['stdout']
        self.stderr = daemon_conf['stderr']
        self.pidfile = daemon_conf['pidfile']

    def daemonize(self):
        """
        do the UNIX double-fork magic, see Stevens' "Advanced
        Programming in the UNIX Environment" for details (ISBN 0201563177)
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        """
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        # decouple from parent environment (Move to EasyApplication.load_config(), delete by leon 20151108)
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                sys.exit(0)
        except OSError, e:
            sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
            sys.exit(1)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = file(self.stdin, 'r')
        so = file(self.stdout, 'a+')
        se = file(self.stderr, 'a+', 0)
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())
        os.umask(022)

        # write pidfile
        atexit.register(self.delpid)
        pid = str(os.getpid())
        file(self.pidfile, 'w+').write("%s\n" % pid)

        # update proc name
        import setproctitle
        setproctitle.setproctitle(self.daemon_conf['name'])

    def delpid(self):
        os.remove(self.pidfile)

    def start(self):
        """
        Start the daemon
        """
        print "\033[32mStarting...\033[0m "

        # Check for a pidfile to see if the daemon already runs
        if not os.path.isfile(self.pidfile):
            pid = None
        else:
            with open(self.pidfile, 'r') as pf:
                pid = pf.read().strip()

        if pid:
            message = "pidfile %s already exist. Daemon already running?\n"
            sys.stderr.write(message % self.pidfile)
            sys.exit(1)

        if os.environ.get('EASY_DEBUG') == '1':
            print "\033[32mReady for debug...\033[0m"
        elif platform.system() != 'Windows':
            print "\033[32mSwitch to background...\033[0m"
            self.daemonize()
        self.run()

    def stop(self):
        """
        Stop the daemon
        """
        # Get the pid from the pidfile

        print "\033[31mStopping...\033[0m "

        try:
            pf = file(self.pidfile, 'r')
            pid = int(pf.read().strip())
            pf.close()
        except IOError:
            pid = None

        if not pid:
            message = "pidfile %s does not exist. Daemon not running?\n"
            sys.stderr.write(message % self.pidfile)
            return  # not an error in a restart

        # Try killing the daemon process
        try:
            for i in range(50):
                os.kill(pid, SIGTERM)
                time.sleep(0.1)
        except OSError, err:
            err = str(err)
            if err.find("No such process") > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                print str(err)
                sys.exit(1)

    def restart(self):
        """
        Restart the daemon
        """
        print "\033[31mRestarting...\033[0m "
        self.stop()
        self.start()

    def process(self, cmd):
        if 'start' == cmd:
            self.start()
        elif 'stop' == cmd:
            self.stop()
        elif 'restart' == cmd:
            self.restart()
        else:
            print "\033[31mUnknown command:\033[0m ", cmd
            sys.exit(2)
        sys.exit(0)

    def before_run(self):
        return

    def run(self):
        return
