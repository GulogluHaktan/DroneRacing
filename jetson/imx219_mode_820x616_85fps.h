/*
 * IMX219: 820x616 RAW10, full field of view (x4 binning of the 3280x2464 array), about 85 fps.
 * For the NVIDIA out-of-tree driver nv_imx219.c (JetPack 6, nvidia-oot/drivers/media/i2c/imx219_mode_tbls.h).
 *
 * Timing (same PLL as the stock modes: pixel rate 182.4 MHz, line length 3448 px -> 18.90 us per line):
 *   frame length 620 lines (616 active + 4 blanking) -> 11.72 ms -> 85.3 fps.
 *   Rolling-shutter readout of the active rows: 616 x 18.90 us = 11.6 ms (estimate; measure it with led_latency.py).
 * The KAIST A2RL S2 system reports 820x616 at 85 Hz on the same sensor; MonoRace reports 90 Hz with a modified driver.
 *
 * NOT TESTED ON HARDWARE. Values follow the IMX219 register map (datasheet) and the mainline Linux imx219.c common registers.
 * Check them against the 1640x1232 table of your driver before use: keep its PLL / CSI values if they differ.
 */
#ifndef __IMX219_MODE_820X616_85FPS_H__
#define __IMX219_MODE_820X616_85FPS_H__

/* imx219_reg = { addr, val } as in imx219_mode_tbls.h; IMX219_TABLE_END ends the list */
static imx219_reg imx219_mode_820x616_85fps[] = {
	/* frame length lines = 620 (0x026C) */
	{0x0160, 0x02},
	{0x0161, 0x6C},
	/* line length pck = 3448 (0x0D78) */
	{0x0162, 0x0D},
	{0x0163, 0x78},
	/* full array: x 0..3279, y 0..2463 */
	{0x0164, 0x00},
	{0x0165, 0x00},
	{0x0166, 0x0C},
	{0x0167, 0xCF},
	{0x0168, 0x00},
	{0x0169, 0x00},
	{0x016A, 0x09},
	{0x016B, 0x9F},
	/* output size 820 x 616 */
	{0x016C, 0x03},
	{0x016D, 0x34},
	{0x016E, 0x02},
	{0x016F, 0x68},
	/* no sub-sampling: x/y odd increment 1 */
	{0x0170, 0x01},
	{0x0171, 0x01},
	/* binning: 0x02 = x4 (H and V) */
	{0x0174, 0x02},
	{0x0175, 0x02},
	/* RAW10 output */
	{0x018C, 0x0A},
	{0x018D, 0x0A},
	{IMX219_TABLE_END, 0x00}
};

#endif
