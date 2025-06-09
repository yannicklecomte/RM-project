import lgpio
import time
import threading

class HX711:
    def __init__(self, dout, pd_sck, gain=128):
        self.PD_SCK = pd_sck
        self.DOUT = dout

        # Mutex for safe multithreading
        self.readLock = threading.Lock()

        # Open GPIO chip
        self.chip = lgpio.gpiochip_open(0)

        # Set GPIO modes
        lgpio.gpio_claim_output(self.chip, self.PD_SCK)
        lgpio.gpio_claim_input(self.chip, self.DOUT)

        self.GAIN = 0
        self.REFERENCE_UNIT = 1
        self.REFERENCE_UNIT_B = 1
        self.OFFSET = 1
        self.OFFSET_B = 1
        self.lastVal = int(0)

        self.DEBUG_PRINTING = False
        self.byte_format = 'MSB'
        self.bit_format = 'MSB'

        self.set_gain(gain)
        time.sleep(1)
        
    def get_gain(self):
      if self.GAIN == 1:
          return 128
      elif self.GAIN == 3:
          return 64
      elif self.GAIN == 2:
          return 32
      return 0  # Default case (should never happen)


    def convertFromTwosComplement24bit(self, inputValue):
        return -(inputValue & 0x800000) + (inputValue & 0x7fffff)

    def is_ready(self):
        return lgpio.gpio_read(self.chip, self.DOUT) == 0

    def set_gain(self, gain):
        if gain == 128:
            self.GAIN = 1
        elif gain == 64:
            self.GAIN = 3
        elif gain == 32:
            self.GAIN = 2

        lgpio.gpio_write(self.chip, self.PD_SCK, 0)
        self.readRawBytes()

    def readNextBit(self):
        lgpio.gpio_write(self.chip, self.PD_SCK, 1)
        time.sleep(0.000001)  # Small delay for stability
        lgpio.gpio_write(self.chip, self.PD_SCK, 0)
        return lgpio.gpio_read(self.chip, self.DOUT)

    def readNextByte(self):
        byteValue = 0
        for _ in range(8):
            if self.bit_format == 'MSB':
                byteValue <<= 1
                byteValue |= self.readNextBit()
            else:
                byteValue >>= 1
                byteValue |= self.readNextBit() * 0x80
        return byteValue 

    def readRawBytes(self):
        self.readLock.acquire()

        while not self.is_ready():
            pass

        firstByte = self.readNextByte()
        secondByte = self.readNextByte()
        thirdByte = self.readNextByte()

        for _ in range(self.GAIN):
            self.readNextBit()

        self.readLock.release()

        if self.byte_format == 'LSB':
            return [thirdByte, secondByte, firstByte]
        else:
            return [firstByte, secondByte, thirdByte]

    def read_long(self):
        dataBytes = self.readRawBytes()
        twosComplementValue = ((dataBytes[0] << 16) |
                               (dataBytes[1] << 8)  |
                               dataBytes[2])

        signedIntValue = self.convertFromTwosComplement24bit(twosComplementValue)
        self.lastVal = signedIntValue
        return int(signedIntValue)

    def read_average(self, times=3):
        if times <= 0:
            raise ValueError("read_average() requires times >= 1!")

        if times == 1:
            return self.read_long()

        valueList = [self.read_long() for _ in range(times)]
        valueList.sort()

        trimAmount = int(len(valueList) * 0.2)
        valueList = valueList[trimAmount:-trimAmount]
        return sum(valueList) / len(valueList)

    def read_median(self, times=3):
        if times <= 0:
            raise ValueError("read_median() requires times > 0!")
      
        if times == 1:
            return self.read_long()

        valueList = [self.read_long() for _ in range(times)]
        valueList.sort()

        midpoint = len(valueList) // 2
        if times % 2 == 1:
            return valueList[midpoint]
        else:
            return sum(valueList[midpoint:midpoint+2]) / 2.0

    def get_value(self, times=3):
        return self.get_value_A(times)

    def get_value_A(self, times=3):
        return self.read_median(times) - self.get_offset_A()

    def get_weight(self, times=3):
        value = self.get_value_A(times)
        return value / self.REFERENCE_UNIT

    def tare(self, times=15):
        backupReferenceUnit = self.get_reference_unit_A()
        self.set_reference_unit_A(1)

        value = self.read_average(times)
        self.set_offset_A(value)
        self.set_reference_unit_A(backupReferenceUnit)

        return value

    def set_offset_A(self, offset):
        self.OFFSET = offset

    def get_offset_A(self):
        return self.OFFSET

    def set_reference_unit_A(self, reference_unit):
        if reference_unit == 0:
            raise ValueError("Reference unit cannot be zero!")
        self.REFERENCE_UNIT = reference_unit

    def get_reference_unit_A(self):
        return self.REFERENCE_UNIT

    def power_down(self):
        self.readLock.acquire()
        lgpio.gpio_write(self.chip, self.PD_SCK, 0)
        lgpio.gpio_write(self.chip, self.PD_SCK, 1)
        time.sleep(0.0001)
        self.readLock.release()           

    def power_up(self):
        self.readLock.acquire()
        lgpio.gpio_write(self.chip, self.PD_SCK, 0)
        time.sleep(0.0001)
        self.readLock.release()

        if self.get_gain() != 128:
            self.readRawBytes()

    def reset(self):
        self.power_down()
        self.power_up()

    def close(self):
        lgpio.gpiochip_close(self.chip)

