import base64
import os
import uuid

import aiohttp

import astrbot.api.message_components as Comp
from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star, register


@register("AstrBot_Plugins_Canvas", "长安某", "gemini 画图工具", "1.2.5")
class GeminiImageGenerator(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        logger.info(f"插件配置加载成功: {self.config}")

        self.api_keys = self.config.get("gemini_api_keys", [])
        self.current_key_index = 0

        self.model_name = self.config.get("gemini_model", "gemini-2.5-flash-image")

        self.image_resolution = self.config.get("image_resolution", "1K")

        logger.info(f"当前使用的 Gemini 模型: {self.model_name}")
        logger.info(f"目标分辨率 (仅Gemini 3.0生效): {self.image_resolution}")

        plugin_dir = os.path.dirname(__file__)
        self.save_dir = os.path.join(plugin_dir, "temp_images")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            logger.info(f"已创建图片临时目录: {self.save_dir}")

        self.api_base_url = self.config.get(
            "api_base_url", "https://generativelanguage.googleapis.com"
        )

        if not self.api_keys:
            logger.error("未配置任何 Gemini API 密钥，请在插件配置中填写")

    def _get_current_api_key(self):
        if not self.api_keys:
            return None
        return self.api_keys[self.current_key_index]

    def _switch_next_api_key(self):
        if not self.api_keys:
            return
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        logger.info(f"已切换到下一个 API 密钥（索引：{self.current_key_index}）")

    @filter.command("gemini_image", alias={"文生图"})
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result(
                "请输入图片描述（示例：/gemini_image 一只戴帽子的猫在月球上）"
            )
            return

        save_path = None

        try:
            yield event.plain_result(f"正在使用 {self.model_name} 生成图片，请稍等...")
            image_data = await self._generate_image_with_retry(prompt)

            if not image_data:
                logger.error("生成失败：所有 API 密钥均尝试完毕")
                yield event.plain_result("生成失败：所有 API 密钥均尝试失败")
                return

            file_name = f"{uuid.uuid4()}.png"
            save_path = os.path.join(self.save_dir, file_name)

            with open(save_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片已保存至: {save_path}")

            yield event.chain_result([Image.fromFileSystem(save_path)])
            logger.info(f"图片发送成功，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片处理失败: {str(e)}")
            yield event.plain_result(f"生成失败: {str(e)}")

        finally:
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除临时图片: {save_path}")
                except Exception as e:
                    logger.warning(f"删除临时图片失败: {str(e)}")

    @filter.command("gemini_edit", alias={"图编辑"})
    async def edit_image(self, event: AstrMessageEvent, prompt: str):
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        image_paths = await self._extract_target_images(event)
        if not image_paths:
            yield event.plain_result("未找到图片，请发送图片并带指令，或回复图片")
            return

        if not prompt.strip():
            yield event.plain_result("请提供编辑描述")
            return

        is_gemini_3 = "gemini-3" in self.model_name

        if len(image_paths) > 1:
            if is_gemini_3:
                logger.info(
                    f"检测到 Gemini 3.0 模型，将使用全部 {len(image_paths)} 张图片作为输入"
                )
            else:
                logger.info(
                    f"当前模型 {self.model_name} 不支持多图，仅使用最后一张图片"
                )
                image_paths = [image_paths[-1]]

        async for result in self._process_image_edit(event, prompt, image_paths):
            yield result

    @filter.llm_tool(name="edit_image")
    async def edit_image_tool(self, event: AstrMessageEvent, prompt: str):
        """编辑图片工具。当用户在消息中发送了图片（一张或多张）、或者引用了图片，并要求修改、改图、润色、转换风格或进行编辑时，必须使用此工具。
        注意：如果用户提供了图片，千万不要使用 generate_image，一定要使用 edit_image。

        Args:
            prompt(string): 编辑指令描述（例如：把图片里的猫改成黑色、变成动漫风格）
        """
        if not self.api_keys:
            yield event.plain_result("错误：未配置任何 Gemini API 密钥")
            return

        if not prompt.strip():
            yield event.plain_result("请提供编辑描述")
            return

        image_paths = await self._extract_target_images(event)
        if not image_paths:
            yield event.plain_result(
                "未找到图片，请在发送消息时附带图片，或回复一张图片"
            )
            return

        is_gemini_3 = "gemini-3" in self.model_name
        if len(image_paths) > 1 and not is_gemini_3:
            image_paths = [image_paths[-1]]

        async for result in self._process_image_edit(event, prompt, image_paths):
            yield result

    @filter.llm_tool(name="generate_image")
    async def generate_image_tool(self, event: AstrMessageEvent, prompt: str):
        """文生图工具。仅在用户想要“凭空生成”一张新图片，且没有提供任何参考图片时使用。
        如果用户发送了图片并要求处理，请务必使用 edit_image 工具，不要使用本工具。

        Args:
            prompt(string): 图片画面描述（例如：画一只在月球上的猫）
        """
        async for result in self.generate_image(event, prompt):
            yield result

    async def _extract_target_images(self, event: AstrMessageEvent) -> list[str]:
        image_paths = []
        try:
            message_components = event.message_obj.message

            for comp in message_components:
                if isinstance(comp, Comp.Image):
                    path = await comp.convert_to_file_path()
                    image_paths.append(path)
                    logger.info(f"提取到当前消息图片: {path}")

            if not image_paths:
                reply_component = None
                for comp in message_components:
                    if isinstance(comp, Comp.Reply):
                        reply_component = comp
                        break

                if reply_component:
                    for quoted_comp in reply_component.chain:
                        if isinstance(quoted_comp, Comp.Image):
                            path = await quoted_comp.convert_to_file_path()
                            image_paths.append(path)
                            logger.info(f"提取到回复图片: {path}")

            return image_paths

        except Exception as e:
            logger.error(f"提取图片失败: {str(e)}", exc_info=True)
            return []

    async def _process_image_edit(
        self, event: AstrMessageEvent, prompt: str, image_paths: list[str]
    ):
        save_path = None
        try:
            yield event.plain_result(
                f"正在使用 {self.model_name} 编辑 {len(image_paths)} 张图片，请稍等..."
            )

            image_data = await self._edit_image_with_retry(prompt, image_paths)

            if not image_data:
                yield event.plain_result("编辑失败：所有 API 密钥均尝试失败")
                return

            save_path = os.path.join(self.save_dir, f"{uuid.uuid4()}_edited.png")
            with open(save_path, "wb") as f:
                f.write(image_data)

            yield event.chain_result([Comp.Image.fromFileSystem(save_path)])
            logger.info(f"图片编辑完成并发送，提示词: {prompt}")

        except Exception as e:
            logger.error(f"图片编辑出错：{str(e)}")
            yield event.plain_result(f"图片编辑失败：{str(e)}")

        finally:
            for path in image_paths:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.warning(f"删除原始图片失败：{str(e)}")

            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info(f"已删除编辑图临时文件：{save_path}")
                except Exception as e:
                    logger.warning(f"删除编辑图失败：{str(e)}")

    async def _edit_image_with_retry(self, prompt, image_paths):
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试编辑图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._edit_image_manually(prompt, image_paths, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有 API 密钥均尝试失败")

        return None

    async def _generate_image_with_retry(self, prompt):
        max_attempts = len(self.api_keys)
        attempts = 0

        while attempts < max_attempts:
            current_key = self._get_current_api_key()
            if not current_key:
                break

            logger.info(
                f"尝试生成图片（密钥索引：{self.current_key_index}，尝试次数：{attempts + 1}/{max_attempts}）"
            )

            try:
                return await self._generate_image_manually(prompt, current_key)
            except Exception as e:
                attempts += 1
                logger.error(f"第{attempts}次尝试失败：{str(e)}")
                if attempts < max_attempts:
                    self._switch_next_api_key()
                else:
                    logger.error("所有 API 密钥均尝试失败")

        return None

    async def _edit_image_manually(self, prompt, image_paths: list[str], api_key):
        model_name = self.model_name

        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.info(f"异步请求地址：{endpoint} | 图片数量: {len(image_paths)}")

        headers = {"Content-Type": "application/json"}

        parts = []
        parts.append({"text": prompt})

        for path in image_paths:
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
                    image_base64 = (
                        base64.b64encode(image_bytes)
                        .decode("utf-8")
                        .replace("\n", "")
                        .replace("\r", "")
                    )
                    parts.append(
                        {"inlineData": {"mimeType": "image/png", "data": image_base64}}
                    )
            except Exception as e:
                logger.error(f"处理图片 {path} 失败: {e}")
                raise e

        gen_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.8,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 1024,
        }

        if "gemini-3" in model_name:
            gen_config["imageConfig"] = {"imageSize": self.image_resolution}

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": gen_config,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f"API 编辑请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    data = await response.json()
            except Exception as e:
                logger.error(f"异步编辑请求失败: {str(e)}")
                raise

        image_data = None
        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    break

        if not image_data:
            error_text = ""
            try:
                for part in data["candidates"][0]["content"]["parts"]:
                    if "text" in part:
                        error_text += part["text"]
            except:
                pass

            error_msg = (
                f"未获取到图片数据。模型回复：{error_text}"
                if error_text
                else "编辑图片成功，但未获取到图片数据"
            )
            raise Exception(error_msg)

        return image_data

    async def _generate_image_manually(self, prompt, api_key):
        model_name = self.model_name

        base_url = self.api_base_url.strip()
        if not base_url.startswith("https://"):
            base_url = f"https://{base_url}"
        if base_url.endswith("/"):
            base_url = base_url[:-1]

        endpoint = (
            f"{base_url}/v1beta/models/{model_name}:generateContent?key={api_key}"
        )
        logger.info(f"异步请求地址：{endpoint}")

        headers = {"Content-Type": "application/json"}

        gen_config = {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.8,
            "topP": 0.95,
            "topK": 40,
            "maxOutputTokens": 1024,
        }

        if "gemini-3" in model_name:
            gen_config["imageConfig"] = {"imageSize": self.image_resolution}
            logger.debug(f"已应用 Gemini 3.0 图片配置: {gen_config['imageConfig']}")

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": gen_config,
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url=endpoint, json=payload, headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(
                            f"API 生成请求失败: HTTP {response.status}, 响应: {response_text}"
                        )
                        response.raise_for_status()

                    data = await response.json()
            except Exception as e:
                logger.error(f"异步生成请求失败: {str(e)}")
                raise

        image_data = None
        if "candidates" in data and len(data["candidates"]) > 0:
            for part in data["candidates"][0]["content"]["parts"]:
                if "inlineData" in part and "data" in part["inlineData"]:
                    base64_str = (
                        part["inlineData"]["data"].replace("\n", "").replace("\r", "")
                    )
                    image_data = base64.b64decode(base64_str)
                    break

        if not image_data:
            error_text = ""
            try:
                for part in data["candidates"][0]["content"]["parts"]:
                    if "text" in part:
                        error_text += part["text"]
            except:
                pass

            error_msg = (
                f"未获取到图片数据。模型回复：{error_text}"
                if error_text
                else "生成图片成功，但未获取到图片数据"
            )
            raise Exception(error_msg)

        return image_data

    async def terminate(self):
        if os.path.exists(self.save_dir):
            try:
                for file in os.listdir(self.save_dir):
                    os.remove(os.path.join(self.save_dir, file))
                os.rmdir(self.save_dir)
                logger.info(f"插件卸载：已清理临时目录 {self.save_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {str(e)}")
        logger.info("Gemini 文生图插件已停用")
