#!/usr/bin/env python3

import re
import smbus


class I2C_Connection:
    """
    Simple I2C communication class compatible with Python 3,
    based on Adafruit's original helper for Raspberry Pi.
    """

    @staticmethod
    def getPiRevision():
        """
        Gets the Raspberry Pi board revision.
        Returns:
            1 if revision is old style (e.g. 0002, 0003), else 2
        """
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    match = re.match(r'Revision\s+:\s+.*(\w{4})$', line)
                    if match:
                        rev = match.group(1)
                        if rev in ['0000', '0002', '0003']:
                            return 1
                        return 2
        except Exception:
            return 0  # Unknown, fallback

    @staticmethod
    def getPiI2CBusNumber():
        """
        Returns the correct I2C bus number for the Pi.
        """
        return 1 if I2C_Connection.getPiRevision() > 1 else 0

    def __init__(self, address, busnum=-1, debug=False):
        self.address = address
        self.debug = debug
        bus_number = busnum if busnum >= 0 else self.getPiI2CBusNumber()
        self.bus = smbus.SMBus(bus_number)

    def reverseByteOrder(self, data):
        """
        Reverses byte order of a multi-byte integer.
        Example: 0x1234 -> 0x3412
        """
        byte_count = len(hex(data)[2:].replace('L', '')[::2])
        val = 0
        for _ in range(byte_count):
            val = (val << 8) | (data & 0xFF)
            data >>= 8
        return val

    def errMsg(self, err=None):
        if self.debug:
            print(f"Error accessing I2C address 0x{self.address:02X}: {err}")
        return -1

    def write8(self, reg, value):
        try:
            self.bus.write_byte_data(self.address, reg, value)
            if self.debug:
                print(f"I2C: Wrote 0x{value:02X} to register 0x{reg:02X}")
        except IOError as err:
            return self.errMsg(err)

    def write16(self, reg, value):
        try:
            self.bus.write_word_data(self.address, reg, value)
            if self.debug:
                print(f"I2C: Wrote 0x{value:04X} to register pair 0x{reg:02X}/0x{reg+1:02X}")
        except IOError as err:
            return self.errMsg(err)

    def writeRaw8(self, value):
        try:
            self.bus.write_byte(self.address, value)
            if self.debug:
                print(f"I2C: Wrote raw 0x{value:02X}")
        except IOError as err:
            return self.errMsg(err)

    def writeList(self, reg, data_list):
        try:
            self.bus.write_i2c_block_data(self.address, reg, data_list)
            if self.debug:
                print(f"I2C: Wrote list to 0x{reg:02X}: {data_list}")
        except IOError as err:
            return self.errMsg(err)

    def readList(self, reg, length):
        try:
            results = self.bus.read_i2c_block_data(self.address, reg, length)
            if self.debug:
                print(f"I2C: Read from 0x{reg:02X}: {results}")
            return results
        except IOError as err:
            return self.errMsg(err)

    def readU8(self, reg):
        try:
            result = self.bus.read_byte_data(self.address, reg)
            if self.debug:
                print(f"I2C: Read U8 0x{result:02X} from 0x{reg:02X}")
            return result
        except IOError as err:
            return self.errMsg(err)

    def readS8(self, reg):
        try:
            result = self.bus.read_byte_data(self.address, reg)
            if result > 127:
                result -= 256
            if self.debug:
                print(f"I2C: Read S8 {result} from 0x{reg:02X}")
            return result
        except IOError as err:
            return self.errMsg(err)

    def readU16(self, reg, little_endian=True):
        try:
            result = self.bus.read_word_data(self.address, reg)
            if not little_endian:
                result = ((result << 8) & 0xFF00) + (result >> 8)
            if self.debug:
                print(f"I2C: Read U16 0x{result:04X} from 0x{reg:02X}")
            return result
        except IOError as err:
            return self.errMsg(err)

    def readS16(self, reg, little_endian=True):
        try:
            result = self.readU16(reg, little_endian)
            if result > 32767:
                result -= 65536
            return result
        except IOError as err:
            return self.errMsg(err)