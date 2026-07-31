/* qemu's mps2-an385 (Cortex-M3) and mps2-an386 (Cortex-M4) machines: 4M of
 * flash at 0x0 and 4M of RAM at 0x20000000. The parity scratch is stack, and
 * the worst single check is a quarter of a megabyte, so 4M is headroom, not a
 * fit. */
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM   : ORIGIN = 0x20000000, LENGTH = 4M
}
